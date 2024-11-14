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

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include "libgccjit.h"
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
  PointerType getVoidPtrType() const {
    return PointerType::get(this->getContext(),
                            VoidType::get(this->getContext()));
  }
  Value createIndexAttrConstant(OpBuilder &builder, Location loc,
                                Type resultType, int64_t value) const;
  Value getSizeInBytes(Location loc, Type type,
                       ConversionPatternRewriter &rewriter) const;
  Value getAlignInBytes(Location loc, Type type,
                        ConversionPatternRewriter &rewriter) const;
  PointerType getElementPtrType(MemRefType type) const;

  void getMemRefDescriptorSizes(Location loc, MemRefType memRefType,
                                ValueRange dynamicSizes,
                                ConversionPatternRewriter &rewriter,
                                SmallVectorImpl<Value> &sizes,
                                SmallVectorImpl<Value> &strides, Value &size,
                                bool sizeInBytes) const;

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

template <typename OpType>
class AllocationLowering : public GCCJITLoweringPattern<OpType> {
protected:
  /// Computes the aligned value for 'input' as follows:
  ///   bumped = input + alignement - 1
  ///   aligned = bumped - bumped % alignment
  Value createAligned(ConversionPatternRewriter &rewriter, Location loc,
                      Value input, Value alignment) const;

  MemRefType getMemRefResultType(OpType op) const;

  Value getAlignment(ConversionPatternRewriter &rewriter, Location loc,
                     OpType op) const;

  int64_t alignedAllocationGetAlignment(ConversionPatternRewriter &rewriter,
                                        Location loc, OpType op) const;

  /// Allocates a memory buffer using an aligned allocation method.
  Value allocateBufferAutoAlign(ConversionPatternRewriter &rewriter,
                                Location loc, Value sizeBytes, OpType op,
                                Value alignment) const;

  virtual void allocateBuffer(ConversionPatternRewriter &rewriter, Location loc,
                              Value size, OpType op) const = 0;

private:
  static constexpr uint64_t kMinAlignedAllocAlignment = 16UL;

public:
  using GCCJITLoweringPattern<OpType>::GCCJITLoweringPattern;
  LogicalResult
  matchAndRewrite(OpType op,
                  typename OpConversionPattern<OpType>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
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

template <typename T>
Value GCCJITLoweringPattern<T>::getSizeInBytes(
    Location loc, Type type, ConversionPatternRewriter &rewriter) const {
  Type gccjitType = getTypeConverter()->convertType(type);
  auto indexType = getIndexType();
  return rewriter.create<gccjit::SizeOfOp>(loc, indexType, gccjitType);
}

template <typename T>
Value GCCJITLoweringPattern<T>::getAlignInBytes(
    Location loc, Type type, ConversionPatternRewriter &rewriter) const {
  Type gccjitType = getTypeConverter()->convertType(type);
  auto indexType = getIndexType();
  return rewriter.create<gccjit::AlignOfOp>(loc, indexType, gccjitType);
}

template <typename T>
PointerType GCCJITLoweringPattern<T>::getElementPtrType(MemRefType type) const {
  auto eltTy = getTypeConverter()->convertType(type.getElementType());
  return PointerType::get(this->getContext(), eltTy);
}

template <typename OpType>
MemRefType AllocationLowering<OpType>::getMemRefResultType(OpType op) const {
  return cast<MemRefType>(op->getResult(0).getType());
}

template <typename OpType>
Value AllocationLowering<OpType>::getAlignment(
    ConversionPatternRewriter &rewriter, Location loc, OpType op) const {
  MemRefType memRefType = op.getType();
  Value alignment;
  if (auto alignmentAttr = op.getAlignment()) {
    Type indexType = this->getIndexType();
    alignment =
        this->createIndexAttrConstant(rewriter, loc, indexType, *alignmentAttr);
  } else {
    alignment =
        this->getAlignInBytes(loc, memRefType.getElementType(), rewriter);
  }
  return alignment;
}

template <typename OpType>
Value AllocationLowering<OpType>::createAligned(
    ConversionPatternRewriter &rewriter, Location loc, Value input,
    Value alignment) const {
  Value one =
      this->createIndexAttrConstant(rewriter, loc, alignment.getType(), 1);
  Value bump = rewriter.create<gccjit::BinaryOp>(loc, alignment.getType(),
                                                 BOp::Minus, alignment, one);
  Value bumped = rewriter.create<gccjit::BinaryOp>(loc, alignment.getType(),
                                                   BOp::Plus, input, bump);
  Value mod = rewriter.create<gccjit::BinaryOp>(loc, alignment.getType(),
                                                BOp::Modulo, bumped, alignment);
  return rewriter.create<gccjit::BinaryOp>(loc, alignment.getType(), BOp::Minus,
                                           bumped, mod);
}

template <typename OpType>
Value AllocationLowering<OpType>::allocateBufferAutoAlign(
    ConversionPatternRewriter &rewriter, Location loc, Value sizeBytes,
    OpType op, Value allocAlignment) const {
  MemRefType memRefType = getMemRefResultType(op);
  sizeBytes = createAligned(rewriter, loc, sizeBytes, allocAlignment);
  Type elementPtrType = this->getElementPtrType(memRefType);
  auto result =
      rewriter
          .create<gccjit::CallOp>(
              loc, this->getVoidPtrType(),
              SymbolRefAttr::get(this->getContext(), "aligned_alloc"),
              ValueRange{allocAlignment, sizeBytes},
              /* tailcall */ nullptr, /* builtin */ rewriter.getUnitAttr())
          .getResult();

  return rewriter.create<gccjit::BitCastOp>(loc, elementPtrType, result);
}

[[gnu::used]]
bool isConvertibleAndHasIdentityMaps(MemRefType type,
                                     const GCCJITTypeConverter &typeConverter) {
  if (!typeConverter.convertType(type.getElementType()))
    return false;
  return type.getLayout().isIdentity();
}

template <typename OpType>
void GCCJITLoweringPattern<OpType>::getMemRefDescriptorSizes(
    Location loc, MemRefType memRefType, ValueRange dynamicSizes,
    ConversionPatternRewriter &rewriter, SmallVectorImpl<Value> &sizes,
    SmallVectorImpl<Value> &strides, Value &size, bool sizeInBytes) const {
  assert(
      isConvertibleAndHasIdentityMaps(memRefType, *this->getTypeConverter()) &&
      "layout maps must have been normalized away");
  assert(count(memRefType.getShape(), ShapedType::kDynamic) ==
             static_cast<ssize_t>(dynamicSizes.size()) &&
         "dynamicSizes size doesn't match dynamic sizes count in memref shape");

  sizes.reserve(memRefType.getRank());
  unsigned dynamicIndex = 0;
  Type indexType = getIndexType();
  for (int64_t size : memRefType.getShape()) {
    sizes.push_back(
        size == ShapedType::kDynamic
            ? dynamicSizes[dynamicIndex++]
            : createIndexAttrConstant(rewriter, loc, indexType, size));
  }

  // Strides: iterate sizes in reverse order and multiply.
  int64_t stride = 1;
  Value runningStride = createIndexAttrConstant(rewriter, loc, indexType, 1);
  strides.resize(memRefType.getRank());
  for (auto i = memRefType.getRank(); i-- > 0;) {
    strides[i] = runningStride;

    int64_t staticSize = memRefType.getShape()[i];
    bool useSizeAsStride = stride == 1;
    if (staticSize == ShapedType::kDynamic)
      stride = ShapedType::kDynamic;
    if (stride != ShapedType::kDynamic)
      stride *= staticSize;

    if (useSizeAsStride)
      runningStride = sizes[i];
    else if (stride == ShapedType::kDynamic)
      runningStride = rewriter.create<gccjit::BinaryOp>(
          loc, indexType, BOp::Mult, runningStride, sizes[i]);
    else
      runningStride = createIndexAttrConstant(rewriter, loc, indexType, stride);
  }
  if (sizeInBytes) {
    // Buffer size in bytes.
    Type elementType =
        this->getTypeConverter()->convertType(memRefType.getElementType());
    size = rewriter.create<gccjit::SizeOfOp>(loc, indexType, elementType);
    size = rewriter.create<gccjit::BinaryOp>(loc, indexType, BOp::Mult, size,
                                             runningStride);
  } else {
    size = runningStride;
  }
}

template <typename OpType>
LogicalResult AllocationLowering<OpType>::matchAndRewrite(
    OpType op, typename OpConversionPattern<OpType>::OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  MemRefType memRefType = getMemRefResultType(op);
  if (!isConvertibleAndHasIdentityMaps(memRefType, *this->getTypeConverter()))
    return rewriter.notifyMatchFailure(op, "incompatible memref type");
  auto loc = op->getLoc();
  auto convertedType = this->getTypeConverter()->convertType(memRefType);

  // Get actual sizes of the memref as values: static sizes are constant
  // values and dynamic sizes are passed to 'alloc' as operands.  In case of
  // zero-dimensional memref, assume a scalar (size 1).
  SmallVector<Value, 4> sizes;
  SmallVector<Value, 4> strides;
  Value size;

  this->getMemRefDescriptorSizes(loc, memRefType, adaptor.getOperands(),
                                 rewriter, sizes, strides, size, true);
  auto elementPtrType = this->getElementPtrType(memRefType);
  auto exprBundle = rewriter.create<ExprOp>(op.getLoc(), elementPtrType);
  {
    auto *block = rewriter.createBlock(&exprBundle.getBody());
    rewriter.setInsertionPointToStart(block);
    // Allocate the underlying buffer.
    this->allocateBuffer(rewriter, loc, size, op);
  }
  rewriter.setInsertionPoint(op);

  auto arrayTy = ArrayType::get(rewriter.getContext(), this->getIndexType(),
                                memRefType.getRank());
  auto sizeArr = rewriter.create<gccjit::NewArrayOp>(loc, arrayTy, sizes);
  auto strideArr = rewriter.create<gccjit::NewArrayOp>(loc, arrayTy, strides);
  auto zero =
      this->createIndexAttrConstant(rewriter, loc, this->getIndexType(), 0);
  // Create the MemRef descriptor.
  rewriter.replaceOpWithNewOp<gccjit::NewStructOp>(
      op, convertedType, ArrayRef<int32_t>{0, 1, 2, 3, 4},
      ValueRange{exprBundle, exprBundle, zero, sizeArr, strideArr});

  return success();
}

struct DeallocOpLowering : public GCCJITLoweringPattern<memref::DeallocOp> {
  using GCCJITLoweringPattern<memref::DeallocOp>::GCCJITLoweringPattern;
  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if ([[maybe_unused]] auto unrankedTy =
            llvm::dyn_cast<UnrankedMemRefType>(op.getMemref().getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "unranked memref type not supported");
    }

    Value ptr = getMemRefDescriptor(adaptor.getMemref(),
                                    cast<MemRefType>(op.getMemref().getType()),
                                    rewriter)
                    .getMemRefDescriptorBufferPtr(op.getLoc());
    auto voidPtrValue =
        rewriter.create<gccjit::BitCastOp>(op.getLoc(), getVoidPtrType(), ptr);

    rewriter.replaceOpWithNewOp<gccjit::CallOp>(
        op, Type{}, SymbolRefAttr::get(rewriter.getContext(), "free"),
        ValueRange{voidPtrValue},
        /* tailcall */ nullptr, /* builtin */ rewriter.getUnitAttr());
    return success();
  }
};

struct AllocaOpLowering : public AllocationLowering<memref::AllocaOp> {
  using AllocationLowering<memref::AllocaOp>::AllocationLowering;
  void allocateBuffer(ConversionPatternRewriter &rewriter, Location loc,
                      Value size, memref::AllocaOp op) const override final {
    auto allocaOp = cast<memref::AllocaOp>(op);
    auto elementType =
        typeConverter->convertType(allocaOp.getType().getElementType());

    auto elementPtrType = PointerType::get(rewriter.getContext(), elementType);

    Value alloca;

    if (auto align = op.getAlignment()) {
      auto alignment =
          createIndexAttrConstant(rewriter, loc, getIndexType(), *align * 8);
      alloca =
          rewriter
              .create<CallOp>(loc, getVoidPtrType(),
                              SymbolRefAttr::get(rewriter.getContext(),
                                                 "__builtin_alloca_with_align"),
                              ValueRange{size, alignment},
                              /* tailcall */ nullptr,
                              /* builtin */ rewriter.getUnitAttr())
              .getResult();
    } else {
      alloca = rewriter
                   .create<CallOp>(loc, getVoidPtrType(),
                                   SymbolRefAttr::get(rewriter.getContext(),
                                                      "__builtin_alloca"),
                                   ValueRange{size},
                                   /* tailcall */ nullptr,
                                   /* builtin */ rewriter.getUnitAttr())
                   .getResult();
    }
    alloca = rewriter.create<BitCastOp>(loc, elementPtrType, alloca);
    rewriter.create<ReturnOp>(loc, alloca);
  }
};

struct AllocOpLowering : public AllocationLowering<memref::AllocOp> {
  void allocateBuffer(ConversionPatternRewriter &rewriter, Location loc,
                      Value sizeBytes,
                      memref::AllocOp op) const override final {
    auto result = allocateBufferAutoAlign(rewriter, loc, sizeBytes, op,
                                          getAlignment(rewriter, loc, op));
    rewriter.create<ReturnOp>(loc, result);
  }
  using AllocationLowering<memref::AllocOp>::AllocationLowering;
};

void removeAssumeAlignmentOp(memref::AssumeAlignmentOp op,
                             GCCJITTypeConverter *typeConverter,
                             IRRewriter &rewriter,
                             MutableArrayRef<OpOperand *> replacement) {
  rewriter.setInsertionPoint(op);
  auto memRefType = cast<MemRefType>(op.getMemref().getType());
  auto descriptorType = typeConverter->getMemrefDescriptorType(memRefType);
  auto materializedMemref = typeConverter->materializeTargetConversion(
      rewriter, op.getLoc(), descriptorType, op.getMemref());
  auto ptrType = cast<FieldAttr>(descriptorType.getFields()[0]).getType();
  auto offsetType = cast<FieldAttr>(descriptorType.getFields()[2]).getType();
  auto arrayType = cast<FieldAttr>(descriptorType.getFields()[3]).getType();
  auto exprBundle = rewriter.create<ExprOp>(op->getLoc(), descriptorType);
  auto *block = rewriter.createBlock(&exprBundle.getBody());
  auto voidPtrType = PointerType::get(rewriter.getContext(),
                                      VoidType::get(rewriter.getContext()));
  rewriter.setInsertionPointToStart(block);
  Value allocPtr = rewriter.create<gccjit::AccessFieldOp>(
      op.getLoc(), ptrType, materializedMemref, rewriter.getIndexAttr(0));
  Value alignedPtr = rewriter.create<gccjit::AccessFieldOp>(
      op.getLoc(), ptrType, materializedMemref, rewriter.getIndexAttr(1));
  Value offset = rewriter.create<gccjit::AccessFieldOp>(
      op.getLoc(), offsetType, materializedMemref, rewriter.getIndexAttr(2));
  Value sizes = rewriter.create<gccjit::AccessFieldOp>(
      op.getLoc(), arrayType, materializedMemref, rewriter.getIndexAttr(3));
  Value strides = rewriter.create<gccjit::AccessFieldOp>(
      op.getLoc(), arrayType, materializedMemref, rewriter.getIndexAttr(4));
  alignedPtr =
      rewriter.create<gccjit::BitCastOp>(op.getLoc(), voidPtrType, alignedPtr);
  alignedPtr =
      rewriter
          .create<gccjit::CallOp>(
              op.getLoc(), voidPtrType,
              SymbolRefAttr::get(rewriter.getContext(),
                                 "__builtin_assume_aligned"),
              ValueRange{alignedPtr, offset},
              /* tailcall */ nullptr, /* builtin */ rewriter.getUnitAttr())
          .getResult();
  alignedPtr =
      rewriter.create<gccjit::BitCastOp>(op.getLoc(), ptrType, alignedPtr);
  auto newMemRef = rewriter.create<gccjit::NewStructOp>(
      op.getLoc(), descriptorType, ArrayRef<int32_t>{0, 1, 2, 3, 4},
      ValueRange{allocPtr, alignedPtr, offset, sizes, strides});
  rewriter.create<gccjit::ReturnOp>(op.getLoc(), newMemRef);
  rewriter.setInsertionPoint(op);
  auto srcValue = typeConverter->materializeSourceConversion(
      rewriter, op.getLoc(), memRefType, exprBundle.getResult());
  for (auto &use : replacement)
    use->set(srcValue);
  rewriter.eraseOp(op);
}

void removeAllAssumeAlignmentOps(ModuleOp moduleOp,
                                 GCCJITTypeConverter *typeConverter,
                                 DominanceInfo &domInfo,
                                 llvm::SmallVectorImpl<Operation *> &ops) {
  for (auto &operation : moduleOp.getOps()) {
    if (auto func = dyn_cast<func::FuncOp>(operation)) {
      llvm::DenseMap<memref::AssumeAlignmentOp, llvm::SmallVector<OpOperand *>>
          replacement;
      func.walk([&](memref::AssumeAlignmentOp op) {
        for (auto &use : op.getMemref().getUses()) {
          auto *user = use.getOwner();
          if (isa<memref::AssumeAlignmentOp>(user))
            continue;
          if (domInfo.properlyDominates(op, user))
            replacement[op].push_back(&use);
        }
      });
      IRRewriter rewriter(func.getContext());
      rewriter.startOpModification(func);
      func->walk([&](memref::AssumeAlignmentOp op) {
        removeAssumeAlignmentOp(op, typeConverter, rewriter, replacement[op]);
      });
      rewriter.finalizeOpModification(func);
      domInfo.invalidate(&func.getFunctionBody());
    }
    ops.push_back(&operation);
  }
}

static Type
convertGlobalMemrefTypeToGCCJIT(MemRefType type,
                                const GCCJITTypeConverter &typeConverter) {
  // GCCJIT type for a global memref will be a multi-dimension array. For
  // declarations or uninitialized global memrefs, we can potentially flatten
  // this to a 1D array.
  Type elementType = typeConverter.convertType(type.getElementType());
  Type arrayTy = elementType;
  // Shape has the outermost dim at index 0, so need to walk it backwards
  for (int64_t dim : llvm::reverse(type.getShape()))
    arrayTy = ArrayType::get(type.getContext(), arrayTy, dim);
  return arrayTy;
}

TypedAttr convertArrayElement(TypedAttr attr,
                              const GCCJITTypeConverter &typeConverter) {
  return llvm::TypeSwitch<TypedAttr, TypedAttr>(attr)
      .Case([&](mlir::IntegerAttr attr) {
        return typeConverter.convertIntegerAttr(attr);
      })
      .Case([&](mlir::FloatAttr attr) {
        return typeConverter.convertFloatAttr(attr);
      })
      .Default([&](auto attr) { return attr; });
}

Value createSplatArray(ArrayRef<int64_t> shape, TypedAttr constant,
                       OpBuilder &builder, Location loc,
                       const GCCJITTypeConverter &typeConverter) {
  constant = convertArrayElement(constant, typeConverter);
  auto arrayTy = constant.getType();
  Value result = builder.create<ConstantOp>(loc, constant);
  for (int64_t dim : llvm::reverse(shape)) {
    SmallVector<Value> elements(dim, result);
    arrayTy = ArrayType::get(builder.getContext(), arrayTy, dim);
    result = builder.create<NewArrayOp>(loc, arrayTy, elements);
  }
  return result;
}

Value createInitializerArray(DenseElementsAttr attr, OpBuilder &builder,
                             Location loc,
                             const GCCJITTypeConverter &typeConverter) {
  if (attr.isSplat()) {
    auto shape = attr.getType().getShape();
    return createSplatArray(shape, attr.getSplatValue<TypedAttr>(), builder,
                            loc, typeConverter);
  }
  auto shape = attr.getType().getShape();
  auto type = typeConverter.convertType(attr.getType().getElementType());
  SmallVector<Value> elements;
  for (auto element : attr.getValues<TypedAttr>()) {
    element = convertArrayElement(element, typeConverter);
    elements.push_back(builder.create<ConstantOp>(loc, element));
  }
  for (int64_t dim : llvm::reverse(shape)) {
    SmallVector<Value> newElements;
    type = ArrayType::get(builder.getContext(), type, dim);
    for (size_t i = 0, e = elements.size(); i < e; i += dim) {
      newElements.push_back(builder.create<NewArrayOp>(
          loc, type, ArrayRef(elements).slice(i, dim)));
    }
    elements = std::move(newElements);
  }
  return elements[0];
}

template <typename T>
static LogicalResult
emitInitializerExpr(Type gccjitType, ElementsAttr attr, OpBuilder &builder,
                    const GCCJITTypeConverter &typeConverter, Value &result,
                    T &&reportFailure, Location loc) {
  auto denseAttr = dyn_cast<DenseElementsAttr>(attr);
  if (!denseAttr)
    return reportFailure("only dense elements attributes are supported");
  result = createInitializerArray(denseAttr, builder, loc, typeConverter);
  return success();
}

struct GlobalOpLowering : public GCCJITLoweringPattern<memref::GlobalOp> {
  using GCCJITLoweringPattern::GCCJITLoweringPattern;

  LogicalResult
  matchAndRewrite(memref::GlobalOp global, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = global.getType();
    if (!isConvertibleAndHasIdentityMaps(type, *getTypeConverter()))
      return failure();

    Type arrayTy = convertGlobalMemrefTypeToGCCJIT(type, *getTypeConverter());

    GlbKind kind = global.isPublic() ? GlbKind::Exported : GlbKind::Internal;

    if (global.isExternal())
      kind = GlbKind::Imported;

    GlbKindAttr kindAttr = GlbKindAttr::get(rewriter.getContext(), kind);

    if (type.getMemorySpace())
      return global.emitOpError(
          "memory space is not supported for global memrefs");

    auto arrayLVTy = LValueType::get(rewriter.getContext(), arrayTy);

    auto newGlobal = rewriter.replaceOpWithNewOp<gccjit::GlobalOp>(
        global, kindAttr, /* readonly */ global.getConstantAttr(),
        global.getSymNameAttr(), TypeAttr::get(arrayLVTy),
        /* reg_name */ nullptr, global.getAlignmentAttr(),
        /* tls_model */ nullptr,
        /* link_section */ nullptr, /* visibility */ nullptr,
        /* initializer*/ nullptr);

    if (!global.isExternal() && !global.isUninitialized()) {
      if (auto initializer = global.getInitialValue()) {
        auto *block = rewriter.createBlock(&newGlobal.getRegion());
        rewriter.setInsertionPointToStart(block);
        Value value;
        if (failed(emitInitializerExpr(
                arrayTy, cast<ElementsAttr>(*initializer), rewriter,
                *getTypeConverter(), value,
                [&](const Twine &msg) { return global.emitError(msg); },
                global.getLoc())))
          return failure();
        rewriter.create<gccjit::ReturnOp>(global.getLoc(), value);
      }
    }
    return success();
  }
};

struct GetGlobalOpLowering : public AllocationLowering<memref::GetGlobalOp> {
  using AllocationLowering::AllocationLowering;

  /// Buffer "allocation" for memref.get_global op is getting the address of
  /// the global variable referenced.
  void allocateBuffer(ConversionPatternRewriter &rewriter, Location loc,
                      Value sizeBytes, memref::GetGlobalOp op) const override {
    MemRefType type = cast<MemRefType>(op.getResult().getType());

    Type arrayTy = convertGlobalMemrefTypeToGCCJIT(type, *getTypeConverter());
    Type arrayLVTy = LValueType::get(rewriter.getContext(), arrayTy);
    Type elementType = getTypeConverter()->convertType(type.getElementType());
    PointerType elementPtrType =
        PointerType::get(rewriter.getContext(), elementType);
    auto ptrTy = PointerType::get(rewriter.getContext(), arrayTy);
    auto getGlobal = rewriter.create<gccjit::GetGlobalOp>(
        op->getLoc(), arrayLVTy, op.getNameAttr());
    auto addressOf = rewriter.create<AddrOp>(loc, ptrTy, getGlobal);
    auto elementAddr =
        rewriter.create<BitCastOp>(loc, elementPtrType, addressOf);
    rewriter.create<ReturnOp>(loc, elementAddr);
  }
};

void ConvertMemrefToGCCJITPass::runOnOperation() {
  auto moduleOp = getOperation();
  auto &domInfo = getAnalysis<DominanceInfo>();
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
  patterns.insert<LoadOpLowering, StoreOpLowering, AllocaOpLowering,
                  AllocOpLowering, DeallocOpLowering, GlobalOpLowering,
                  GetGlobalOpLowering>(typeConverter, &getContext());
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<gccjit::GCCJITDialect, BuiltinDialect>();
  target.addIllegalDialect<memref::MemRefDialect>();
  llvm::SmallVector<Operation *> ops;
  removeAllAssumeAlignmentOps(moduleOp, &typeConverter, domInfo, ops);
  if (failed(applyPartialConversion(ops, target, std::move(patterns))))
    signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> mlir::gccjit::createConvertMemrefToGCCJITPass() {
  return std::make_unique<ConvertMemrefToGCCJITPass>();
}
