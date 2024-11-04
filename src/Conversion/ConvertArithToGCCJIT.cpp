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

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Transforms/DialectConversion.h>

#include "mlir-gccjit/Conversion/Conversions.h"
#include "mlir-gccjit/Conversion/TypeConverter.h"
#include "mlir-gccjit/IR/GCCJITAttrs.h"
#include "mlir-gccjit/IR/GCCJITOps.h"
#include "mlir-gccjit/IR/GCCJITOpsEnums.h"
#include "mlir-gccjit/IR/GCCJITTypes.h"
#include "mlir-gccjit/Passes.h"

using namespace mlir;
using namespace mlir::gccjit;

namespace {
struct ConvertArithToGCCJITPass
    : public ConvertArithToGCCJITBase<ConvertArithToGCCJITPass> {
  using ConvertArithToGCCJITBase::ConvertArithToGCCJITBase;
  void runOnOperation() override final;
};

template <typename T>
class GCCJITLoweringPattern : public mlir::OpConversionPattern<T> {
protected:
  const GCCJITTypeConverter *getTypeConverter() const {
    return static_cast<const GCCJITTypeConverter *>(this->typeConverter);
  }

public:
  using OpConversionPattern<T>::OpConversionPattern;
};

class ConstantOpLowering : public GCCJITLoweringPattern<arith::ConstantOp> {
public:
  using GCCJITLoweringPattern::GCCJITLoweringPattern;
  mlir::LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto attr = op.getValue();

    if (auto value = dyn_cast<mlir::IntegerAttr>(attr)) {
      rewriter.replaceOpWithNewOp<gccjit::ConstantOp>(
          op, getTypeConverter()->convertIntegerAttr(value));
      return mlir::success();
    }

    if (auto value = dyn_cast<mlir::FloatAttr>(attr)) {
      rewriter.replaceOpWithNewOp<gccjit::ConstantOp>(
          op, getTypeConverter()->convertFloatAttr(value));
      return mlir::success();
    }

    return mlir::failure();
  }
};

class CmpIOpLowering : public GCCJITLoweringPattern<arith::CmpIOp> {
  void getComparison(gccjit::CmpOp &kind, bool &signedness,
                     arith::CmpIPredicate pred) const {
    signedness = false;
    switch (pred) {
    case arith::CmpIPredicate::eq:
      kind = gccjit::CmpOp::Eq;
      break;
    case arith::CmpIPredicate::ne:
      kind = gccjit::CmpOp::Ne;
      break;

    case arith::CmpIPredicate::slt:
      signedness = true;
      [[fallthrough]];
    case arith::CmpIPredicate::ult:
      kind = gccjit::CmpOp::Lt;
      break;

    case arith::CmpIPredicate::sle:
      signedness = true;
      [[fallthrough]];
    case arith::CmpIPredicate::ule:
      kind = gccjit::CmpOp::Le;
      break;

    case arith::CmpIPredicate::sgt:
      signedness = true;
      [[fallthrough]];
    case arith::CmpIPredicate::ugt:
      kind = gccjit::CmpOp::Gt;
      break;

    case arith::CmpIPredicate::sge:
      signedness = true;
      [[fallthrough]];
    case arith::CmpIPredicate::uge:
      kind = gccjit::CmpOp::Ge;
      break;
    }
  }

public:
  using GCCJITLoweringPattern::GCCJITLoweringPattern;
  mlir::LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto inputTy = cast<IntType>(lhs.getType());
    auto pred = adaptor.getPredicate();
    gccjit::CmpOp kind;
    bool signedness;
    getComparison(kind, signedness, pred);
    auto i1 = getTypeConverter()->convertType(op.getResult().getType());
    if (signedness && !getTypeConverter()->isSigned(inputTy)) {
      auto signedType = getTypeConverter()->makeSigned(inputTy);
      lhs = rewriter.create<gccjit::BitCastOp>(op.getLoc(), signedType, lhs);
      rhs = rewriter.create<gccjit::BitCastOp>(op.getLoc(), signedType, rhs);
    }
    auto cmpAttr = CmpOpAttr::get(op.getContext(), kind);
    rewriter.replaceOpWithNewOp<gccjit::CompareOp>(op, i1, cmpAttr, lhs, rhs);
    return mlir::success();
  }
};
template <class Op, BOp Kind>
class TrivialBinOpConversion : public GCCJITLoweringPattern<Op> {
  using GCCJITLoweringPattern<Op>::GCCJITLoweringPattern;
  mlir::LogicalResult
  matchAndRewrite(Op op, typename GCCJITLoweringPattern<Op>::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto resultTy = lhs.getType();
    auto kind = BOpAttr::get(op.getContext(), Kind);
    rewriter.replaceOpWithNewOp<gccjit::BinaryOp>(op, resultTy, kind, lhs, rhs);
    return mlir::success();
  }
};

using AddIOpLowering = TrivialBinOpConversion<arith::AddIOp, BOp::Plus>;
using AddFOpLowering = TrivialBinOpConversion<arith::AddFOp, BOp::Plus>;
using MulFOpLowering = TrivialBinOpConversion<arith::MulFOp, BOp::Mult>;

void ConvertArithToGCCJITPass::runOnOperation() {
  auto moduleOp = getOperation();
  auto typeConverter = GCCJITTypeConverter();
  // unrealized conversions
  auto materializeAsUnrealizedCast = [](OpBuilder &builder, Type resultType,
                                        ValueRange inputs,
                                        Location loc) -> Value {
    if (inputs.size() != 1)
      return Value();

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  };
  typeConverter.addTargetMaterialization(materializeAsUnrealizedCast);
  typeConverter.addSourceMaterialization(materializeAsUnrealizedCast);
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<ConstantOpLowering, CmpIOpLowering, AddIOpLowering,
               AddFOpLowering, MulFOpLowering>(typeConverter, &getContext());
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<gccjit::GCCJITDialect>();
  target.addIllegalDialect<arith::ArithDialect>();
  llvm::SmallVector<Operation *> ops;
  for (auto func : moduleOp.getOps<func::FuncOp>())
    ops.push_back(func);
  for (auto func : moduleOp.getOps<gccjit::FuncOp>())
    ops.push_back(func);
  if (failed(applyPartialConversion(ops, target, std::move(patterns))))
    signalPassFailure();
}
} // namespace

std::unique_ptr<Pass> mlir::gccjit::createConvertArithToGCCJITPass() {
  return std::make_unique<ConvertArithToGCCJITPass>();
}
