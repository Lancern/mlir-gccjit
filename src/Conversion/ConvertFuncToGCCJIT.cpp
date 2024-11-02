// Copyright 2024 Schrodinger ZHU Yifan <i@zhuyi.fan>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir-gccjit/Conversion/Conversions.h"
#include "mlir-gccjit/Conversion/TypeConverter.h"
#include "mlir-gccjit/IR/GCCJITAttrs.h"
#include "mlir-gccjit/IR/GCCJITOps.h"
#include "mlir-gccjit/IR/GCCJITOpsEnums.h"
#include "mlir-gccjit/IR/GCCJITTypes.h"
#include "mlir-gccjit/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include <llvm-20/llvm/ADT/SmallVector.h>
#include <llvm-20/llvm/Support/raw_ostream.h>

using namespace mlir;
using namespace mlir::gccjit;

namespace {

struct ConvertFuncToGCCJITPass
    : public ConvertFuncToGCCJITBase<ConvertFuncToGCCJITPass> {
  using ConvertFuncToGCCJITBase::ConvertFuncToGCCJITBase;
  void runOnOperation() override final;
};

void ConvertFuncToGCCJITPass::runOnOperation() {
  auto moduleOp = getOperation();
  SymbolTable symbolTable(moduleOp);
  auto typeConverter = GCCJITTypeConverter();
  mlir::RewritePatternSet patterns(&getContext());
  populateFuncToGCCJITPatterns(&getContext(), typeConverter, patterns,
                               symbolTable);
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<gccjit::GCCJITDialect>();
  target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
  target.addIllegalDialect<mlir::func::FuncDialect>();
  ConversionConfig config;
  llvm::DenseSet<Operation *> unlegalizedOps;
  config.unlegalizedOps = &unlegalizedOps;
  config.notifyCallback = [&](Diagnostic &diag) { diag.print(llvm::errs()); };
  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns),
                                    config)))
    signalPassFailure();
}

template <typename T>
class GCCJITLoweringPattern : public mlir::OpConversionPattern<T> {
protected:
  const SymbolTable &symbolTable;

public:
  const GCCJITTypeConverter *getTypeConverter() const {
    return static_cast<const GCCJITTypeConverter *>(this->typeConverter);
  }

  template <typename... Args>
  GCCJITLoweringPattern(const SymbolTable &symbolTable,
                        const GCCJITTypeConverter &typeConverter,
                        Args &&...args)
      : mlir::OpConversionPattern<T>(typeConverter,
                                     std::forward<Args>(args)...),
        symbolTable(symbolTable) {}
};

NewStructOp packValues(mlir::Location loc, mlir::ValueRange values,
                       const GCCJITTypeConverter &typeConverter,
                       mlir::TypeRange types,
                       mlir::ConversionPatternRewriter &rewriter,
                       FunctionType func) {
  auto packedType =
      typeConverter.convertAndPackTypesIfNonSingleton(types, func);
  auto structType = cast<gccjit::StructType>(packedType);
  auto indices =
      llvm::to_vector(llvm::seq<int>(0, structType.getFields().size()));
  auto indicesAttr = rewriter.getDenseI32ArrayAttr(indices);
  auto packedValue = rewriter.create<gccjit::NewStructOp>(loc, structType,
                                                          indicesAttr, values);
  return packedValue;
}

class ReturnOpLowering : public GCCJITLoweringPattern<func::ReturnOp> {
public:
  using GCCJITLoweringPattern::GCCJITLoweringPattern;
  mlir::LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() == 0) {
      rewriter.replaceOpWithNewOp<gccjit::ReturnOp>(op, Value{});
    } else if (op.getNumOperands() == 1) {
      rewriter.replaceOpWithNewOp<gccjit::ReturnOp>(
          op, adaptor.getOperands().front());
    } else {
      auto packed =
          packValues(op.getLoc(), adaptor.getOperands(), *getTypeConverter(),
                     op.getOperandTypes(), rewriter,
                     op->getParentOfType<func::FuncOp>().getFunctionType());
      rewriter.replaceOpWithNewOp<gccjit::ReturnOp>(op, packed);
    }
    return mlir::success();
  }
};

class ConstantOpLowering : public GCCJITLoweringPattern<func::ConstantOp> {
public:
  using GCCJITLoweringPattern::GCCJITLoweringPattern;
  mlir::LogicalResult
  matchAndRewrite(func::ConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto value = op.getValueAttr();
    auto resultTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<gccjit::FnAddrOp>(op, resultTy, value);
    return mlir::success();
  }
};

class CallOpLowering : public GCCJITLoweringPattern<func::CallOp> {
public:
  using GCCJITLoweringPattern::GCCJITLoweringPattern;
  mlir::LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto callee = op.getCalleeAttr();
    auto funcOp = dyn_cast<func::FuncOp>(symbolTable.lookup(callee.getValue()));
    if (!funcOp)
      return mlir::failure();
    Type resultTy = getTypeConverter()->convertAndPackTypesIfNonSingleton(
        op->getResultTypes(), funcOp.getFunctionType());
    auto callOp = rewriter.create<gccjit::CallOp>(op.getLoc(), resultTy, callee,
                                                  adaptor.getOperands());
    if (op->getNumResults() <= 1)
      rewriter.replaceOp(op, callOp);
    else {
      llvm::SmallVector<Value> unpacked;
      for (auto [idx, resultTy] : llvm::enumerate(op->getResultTypes())) {
        auto convertedType = getTypeConverter()->convertType(resultTy);
        unpacked.push_back(rewriter.create<gccjit::AccessFieldOp>(
            op.getLoc(), convertedType, callOp.getResult(),
            rewriter.getIndexAttr(idx)));
      }
      rewriter.replaceOp(op, unpacked);
    }
    return mlir::success();
  }
};

class CallIndirectOpLowering
    : public GCCJITLoweringPattern<func::CallIndirectOp> {
public:
  using GCCJITLoweringPattern::GCCJITLoweringPattern;
  mlir::LogicalResult
  matchAndRewrite(func::CallIndirectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertAndPackTypesIfNonSingleton(
        op->getResultTypes(), op.getCallee().getType());
    auto callOp = rewriter.create<gccjit::PtrCallOp>(
        op.getLoc(), resultTy, adaptor.getCallee(), adaptor.getOperands());
    if (op->getNumResults() <= 1)
      rewriter.replaceOp(op, callOp);
    else {
      llvm::SmallVector<Value> unpacked;
      for (auto [idx, resultTy] : llvm::enumerate(op->getResultTypes())) {
        auto convertedType = getTypeConverter()->convertType(resultTy);
        unpacked.push_back(rewriter.create<gccjit::AccessFieldOp>(
            op.getLoc(), convertedType, callOp.getResult(),
            rewriter.getIndexAttr(idx)));
      }
      rewriter.replaceOp(op, unpacked);
    }
    return mlir::success();
  }
};

class FuncOpLowering : public GCCJITLoweringPattern<func::FuncOp> {
private:
  FnKindAttr getFnKindAttr(func::FuncOp op) const {
    auto visibility = op.getVisibility();
    switch (visibility) {
    case SymbolTable::Visibility::Public:
      return FnKindAttr::get(op.getContext(), FnKind::Exported);
    case SymbolTable::Visibility::Private:
      return FnKindAttr::get(op.getContext(), op.getFunctionBody().empty()
                                                  ? FnKind::Imported
                                                  : FnKind::Internal);
    case SymbolTable::Visibility::Nested:
      return FnKindAttr::get(op.getContext(), FnKind::Imported);
    }
  }

  void convertEntryBlockArguments(
      Block *blk, mlir::ConversionPatternRewriter &rewriter,
      llvm::DenseMap<BlockArgument, Value> argMap) const {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(blk);
    if (blk->getNumArguments() == 0)
      return;
    llvm::SmallVector<Type> argTypes;
    for (auto blkArg : blk->getArguments()) {
      auto originalTy = blkArg.getType();
      auto convertedTy = getTypeConverter()->convertType(originalTy);
      auto varTy = LValueType::get(getContext(), convertedTy);
      argMap[blkArg] = blkArg;
      blkArg.setType(varTy);
      auto loaded =
          rewriter.create<AsRValueOp>(blkArg.getLoc(), convertedTy, blkArg);
      auto coercion = rewriter.create<UnrealizedConversionCastOp>(
          blkArg.getLoc(), originalTy, loaded.getResult());
      TypeConverter::SignatureConversion conversion;
    }
    rewriter.applySignatureConversion(blk, conversion);
  }

public:
  using GCCJITLoweringPattern::GCCJITLoweringPattern;
  mlir::LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto funcType = op.getFunctionType();
    auto convertedType =
        getTypeConverter()->convertFunctionType(funcType, false);
    auto kind = getFnKindAttr(op);
    auto funcOp = rewriter.create<gccjit::FuncOp>(
        op.getLoc(), op.getSymNameAttr(), kind, TypeAttr::get(convertedType),
        ArrayAttr::get(getContext(), {}));

    if (!op.getFunctionBody().empty()) {
      llvm::DenseMap<BlockArgument, Value> argMap;
      rewriter.inlineRegionBefore(op.getFunctionBody(), funcOp.getBody(),
                                  funcOp.getBody().end());
      if (funcOp.getBody().getNumArguments() != 0)
        convertEntryBlockArguments(&funcOp.getBody().front(), rewriter, argMap);
    }

    rewriter.eraseOp(op);

    return mlir::success();
  }
};

} // namespace

void mlir::gccjit::populateFuncToGCCJITPatterns(
    MLIRContext *context, GCCJITTypeConverter &typeConverter,
    RewritePatternSet &patterns, SymbolTable &symbolTable) {
  patterns.add<ReturnOpLowering, ConstantOpLowering, CallOpLowering,
               CallIndirectOpLowering, FuncOpLowering>(symbolTable,
                                                       typeConverter, context);
}

std::unique_ptr<Pass> mlir::gccjit::createConvertFuncToGCCJITPass() {
  return std::make_unique<ConvertFuncToGCCJITPass>();
}
