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

  IntType getIndexType() const;
  Value createIndexAttrConstant(OpBuilder &builder, Location loc,
                                Type resultType, int64_t value) const;

  class MemRefDescriptor {
  private:
    Value descriptor;
    MemRefType type;

    ConversionPatternRewriter &rewriter;
    const GCCJITLoweringPattern<T> &pattern;

    MemRefDescriptor(Value descriptor, MemRefType type,
                     ConversionPatternRewriter &rewriter,
                     const GCCJITLoweringPattern<T> &pattern);

  public:
    friend class GCCJITLoweringPattern<T>;

    Value getOffset(Location loc) const;

    Value getAlignedPtr(Location loc) const;

    Value getMemRefDescriptorBufferPtr(Location loc) const;

    Value getStridedElementLValue(Location loc, Operation *materializationPoint,
                                  ValueRange indices) const;
  };

  MemRefDescriptor
  getMemRefDescriptor(Value descriptor, MemRefType type,
                      ConversionPatternRewriter &rewriter) const;

public:
  using OpConversionPattern<T>::OpConversionPattern;
};

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
    MemRefDescriptor descriptor =
        getMemRefDescriptor(adaptor.getMemref(), type, rewriter);
    Value dataLValue = descriptor.getStridedElementLValue(op.getLoc(), op,
                                                          adaptor.getIndices());
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
      MemRefDescriptor descriptor =
          getMemRefDescriptor(adaptor.getMemref(), type, rewriter);
      Value dataLValue = descriptor.getStridedElementLValue(
          op->getLoc(), op, adaptor.getIndices());
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

template <typename T> IntType GCCJITLoweringPattern<T>::getIndexType() const {
  return IntType::get(this->getContext(), GCC_JIT_TYPE_SIZE_T);
}

template <typename T>
Value GCCJITLoweringPattern<T>::createIndexAttrConstant(OpBuilder &builder,
                                                        Location loc,
                                                        Type resultType,
                                                        int64_t value) const {
  auto indexTy = getIndexType();
  auto intAttr = IntAttr::get(this->getContext(), indexTy,
                              {64, static_cast<uint64_t>(value)});
  return builder.create<gccjit::ConstantOp>(loc, resultType, intAttr);
}

template <typename T>
GCCJITLoweringPattern<T>::MemRefDescriptor::MemRefDescriptor(
    Value descriptor, MemRefType type, ConversionPatternRewriter &rewriter,
    const GCCJITLoweringPattern<T> &pattern)
    : descriptor(descriptor), type(type), rewriter(rewriter), pattern(pattern) {
}
template <typename T>
Value GCCJITLoweringPattern<T>::MemRefDescriptor::getOffset(
    Location loc) const {
  auto indexTy = pattern.getIndexType();
  return rewriter.create<gccjit::AccessFieldOp>(loc, indexTy, descriptor,
                                                rewriter.getIndexAttr(2));
}

template <typename T>
Value GCCJITLoweringPattern<T>::MemRefDescriptor::getAlignedPtr(
    Location loc) const {
  auto elementType =
      pattern.getTypeConverter()->convertType(type.getElementType());
  auto ptrTy = PointerType::get(pattern.getContext(), elementType);
  return rewriter.create<gccjit::AccessFieldOp>(loc, ptrTy, descriptor,
                                                rewriter.getIndexAttr(1));
}

template <typename T>
Value GCCJITLoweringPattern<T>::MemRefDescriptor::getMemRefDescriptorBufferPtr(
    Location loc) const {
  auto [strides, offsetCst] = getStridesAndOffset(type);
  auto alignedPtr = getAlignedPtr(loc);
  // For zero offsets, we already have the base pointer.
  if (offsetCst == 0)
    return alignedPtr;

  // Otherwise add the offset to the aligned base.
  Type indexType = pattern.getIndexType();
  Value offsetVal = ShapedType::isDynamic(offsetCst)
                        ? getOffset(loc)
                        : pattern.createIndexAttrConstant(rewriter, loc,
                                                          indexType, offsetCst);
  Type elementType =
      pattern.getTypeConverter()->convertType(type.getElementType());
  auto lvalueTy = LValueType::get(rewriter.getContext(), elementType);
  auto lvalue =
      rewriter.create<gccjit::DerefOp>(loc, lvalueTy, alignedPtr, offsetVal);
  return rewriter.create<gccjit::AddrOp>(
      loc, PointerType::get(rewriter.getContext(), elementType), lvalue);
}

template <typename T>
Value GCCJITLoweringPattern<T>::MemRefDescriptor::getStridedElementLValue(
    Location loc, Operation *materializationPoint, ValueRange indices) const {
  Value materializedMemref = nullptr;
  Value ptrToStrideField = nullptr;
  auto [strides, offset] = getStridesAndOffset(type);
  auto indexTy = IntType::get(rewriter.getContext(), GCC_JIT_TYPE_SIZE_T);
  auto elementType =
      pattern.getTypeConverter()->convertType(type.getElementType());
  auto doMaterialization = [&]() {
    if (materializedMemref)
      return;
    OpBuilder::InsertionGuard guard(rewriter);
    if (materializationPoint)
      rewriter.setInsertionPoint(materializationPoint);
    else
      rewriter.setInsertionPointAfter(descriptor.getDefiningOp());
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

  Value base = getMemRefDescriptorBufferPtr(loc);
  Value index;
  for (int i = 0, e = indices.size(); i < e; ++i) {
    Value increment = indices[i];
    if (strides[i] != 1) { // Skip if stride is 1.
      Value stride = ShapedType::isDynamic(strides[i])
                         ? generateStride(i)
                         : pattern.createIndexAttrConstant(rewriter, loc,
                                                           indexTy, strides[i]);
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

template <typename T>
typename GCCJITLoweringPattern<T>::MemRefDescriptor
GCCJITLoweringPattern<T>::getMemRefDescriptor(
    Value descriptor, MemRefType type,
    ConversionPatternRewriter &rewriter) const {
  return {descriptor, type, rewriter, *this};
}

} // namespace

std::unique_ptr<Pass> mlir::gccjit::createConvertMemrefToGCCJITPass() {
  return std::make_unique<ConvertMemrefToGCCJITPass>();
}
