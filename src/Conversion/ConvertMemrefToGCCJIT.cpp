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

#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>

#include "libgccjit.h"
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
struct ConvertMemrefToGCCJITPass
    : public ConvertMemrefToGCCJITBase<ConvertMemrefToGCCJITPass> {
  using ConvertMemrefToGCCJITBase::ConvertMemrefToGCCJITBase;
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

Value createIndexAttrConstant(OpBuilder &builder, Location loc, Type resultType,
                              int64_t value) {

  auto indexTy = IntType::get(builder.getContext(), GCC_JIT_TYPE_SIZE_T);
  auto intAttr = IntAttr::get(builder.getContext(), indexTy,
                              {64, static_cast<uint64_t>(value)});
  return builder.create<gccjit::ConstantOp>(loc, resultType, intAttr);
}

Value getMemRefDescriptorOffset(OpBuilder &builder, Value descriptor,
                                Location loc) {
  auto indexTy = IntType::get(builder.getContext(), GCC_JIT_TYPE_SIZE_T);
  return builder.create<gccjit::AccessFieldOp>(loc, indexTy, descriptor,
                                               builder.getIndexAttr(2));
}

Value getMemRefDiscriptorAlignedPtr(OpBuilder &builder, Value descriptor,
                                    const GCCJITTypeConverter &converter,
                                    Location loc, MemRefType type) {
  auto elementType = converter.convertType(type.getElementType());
  auto ptrTy = PointerType::get(builder.getContext(), elementType);
  return builder.create<gccjit::AccessFieldOp>(loc, ptrTy, descriptor,
                                               builder.getIndexAttr(1));
}

Value getMemRefDescriptorBufferPtr(OpBuilder &builder, Location loc,
                                   Value descriptor,
                                   const GCCJITTypeConverter &converter,
                                   MemRefType type) {
  auto [strides, offsetCst] = getStridesAndOffset(type);
  auto alignedPtr =
      getMemRefDiscriptorAlignedPtr(builder, descriptor, converter, loc, type);

  // For zero offsets, we already have the base pointer.
  if (offsetCst == 0)
    return alignedPtr;

  // Otherwise add the offset to the aligned base.
  Type indexType = IntType::get(builder.getContext(), GCC_JIT_TYPE_SIZE_T);
  Value offsetVal =
      ShapedType::isDynamic(offsetCst)
          ? getMemRefDescriptorOffset(builder, descriptor, loc)
          : createIndexAttrConstant(builder, loc, indexType, offsetCst);
  Type elementType = converter.convertType(type.getElementType());
  auto lvalueTy = LValueType::get(builder.getContext(), elementType);
  auto lvalue =
      builder.create<gccjit::DerefOp>(loc, lvalueTy, alignedPtr, offsetVal);
  return builder.create<gccjit::AddrOp>(
      loc, PointerType::get(builder.getContext(), elementType), lvalue);
}

Value getStridedElementLValue(Location loc, MemRefType type, Value descriptor,
                              ExprOp parent, ValueRange indices,
                              const GCCJITTypeConverter &typeConverter,
                              ConversionPatternRewriter &rewriter) {
  Value materializedMemref = nullptr;
  Value ptrToStrideField = nullptr;
  auto [strides, offset] = getStridesAndOffset(type);
  auto indexTy = IntType::get(rewriter.getContext(), GCC_JIT_TYPE_SIZE_T);
  auto elementType = typeConverter.convertType(type.getElementType());
  auto doMaterialization = [&]() {
    if (materializedMemref)
      return;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(parent);
    auto lvalueTy =
        LValueType::get(rewriter.getContext(), descriptor.getType());
    materializedMemref = rewriter.create<gccjit::LocalOp>(
        loc, lvalueTy, nullptr, nullptr, nullptr);
    rewriter.create<gccjit::AssignOp>(loc, descriptor, materializedMemref);
  };
  auto generateStride = [&](size_t i) -> Value {
    doMaterialization();
    if (!ptrToStrideField) {
      auto descriptorTy = cast<StructType>(descriptor.getType());
      auto fieldTy = cast<ArrayType>(
          cast<FieldAttr>(descriptorTy.getRecordFields()[4]).getType());
      auto fieldLValueTy = LValueType::get(rewriter.getContext(), fieldTy);
      auto strideField = rewriter.create<gccjit::AccessFieldOp>(
          loc, fieldLValueTy, materializedMemref, rewriter.getIndexAttr(4));
      auto ptrToStrideArray = rewriter.create<gccjit::AddrOp>(
          loc, PointerType::get(rewriter.getContext(), fieldTy), strideField);
      ptrToStrideField = rewriter.create<gccjit::BitCastOp>(
          loc, PointerType::get(rewriter.getContext(), indexTy),
          ptrToStrideArray);
    }
    auto offset = rewriter.create<gccjit::AccessFieldOp>(
        loc, indexTy, ptrToStrideField, rewriter.getIndexAttr(i));
    auto strideLValue = rewriter.create<gccjit::DerefOp>(
        loc, LValueType::get(rewriter.getContext(), indexTy), ptrToStrideField,
        offset);
    return rewriter.create<gccjit::AsRValueOp>(loc, indexTy, strideLValue);
  };

  Value base = getMemRefDescriptorBufferPtr(rewriter, loc, descriptor,
                                            typeConverter, type);
  Value index;
  for (int i = 0, e = indices.size(); i < e; ++i) {
    Value increment = indices[i];
    if (strides[i] != 1) { // Skip if stride is 1.
      Value stride =
          ShapedType::isDynamic(strides[i])
              ? generateStride(i)
              : createIndexAttrConstant(rewriter, loc, indexTy, strides[i]);
      increment = rewriter.create<gccjit::BinaryOp>(loc, indexTy, BOp::Mult,
                                                    increment, stride);
    }
    index = index ? rewriter.create<gccjit::BinaryOp>(loc, indexTy, BOp::Plus,
                                                      index, increment)
                  : increment;
  }

  return rewriter.create<gccjit::DerefOp>(
      loc, LValueType::get(rewriter.getContext(), elementType), base, index);
}

class LoadOpLowering : public GCCJITLoweringPattern<memref::LoadOp> {
public:
  using GCCJITLoweringPattern::GCCJITLoweringPattern;
  mlir::LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = op.getMemRefType();
    auto retTy = typeConverter->convertType(op.getResult().getType());
    auto exprBundle = rewriter.replaceOpWithNewOp<ExprOp>(op, retTy);
    auto *block = rewriter.createBlock(&exprBundle.getBody());
    rewriter.setInsertionPointToStart(block);
    Value dataLValue = getStridedElementLValue(
        op.getLoc(), type, adaptor.getMemref(), exprBundle,
        adaptor.getIndices(), *getTypeConverter(), rewriter);
    auto rvalue = rewriter.create<AsRValueOp>(op.getLoc(), retTy, dataLValue);
    rewriter.create<ReturnOp>(op.getLoc(), rvalue);
    return success();
  }
};

class StoreOpLowering : public GCCJITLoweringPattern<memref::StoreOp> {
public:
  using GCCJITLoweringPattern::GCCJITLoweringPattern;
  mlir::LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = op.getMemRefType();
    auto elemTy = typeConverter->convertType(type.getElementType());
    auto elemLValueTy = LValueType::get(rewriter.getContext(), elemTy);
    auto expr = rewriter.create<ExprOp>(op->getLoc(), elemLValueTy, true);
    auto *block = rewriter.createBlock(&expr.getBody());
    {
      rewriter.setInsertionPointToStart(block);
      Value dataLValue = getStridedElementLValue(
          op.getLoc(), type, adaptor.getMemref(), expr, adaptor.getIndices(),
          *getTypeConverter(), rewriter);
      rewriter.create<ReturnOp>(op.getLoc(), dataLValue);
    }
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<AssignOp>(op, adaptor.getValue(), expr);
    return success();
  }
};

void ConvertMemrefToGCCJITPass::runOnOperation() {
  auto moduleOp = getOperation();
  auto typeConverter = GCCJITTypeConverter();
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
  patterns.insert<LoadOpLowering, StoreOpLowering>(typeConverter,
                                                   &getContext());
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<gccjit::GCCJITDialect>();
  target.addIllegalDialect<memref::MemRefDialect>();
  llvm::SmallVector<Operation *> ops;
  for (auto func : moduleOp.getOps<func::FuncOp>())
    ops.push_back(func);
  if (failed(applyPartialConversion(ops, target, std::move(patterns))))
    signalPassFailure();
}
} // namespace

std::unique_ptr<Pass> mlir::gccjit::createConvertMemrefToGCCJITPass() {
  return std::make_unique<ConvertMemrefToGCCJITPass>();
}
