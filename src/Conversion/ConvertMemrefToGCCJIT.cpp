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
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

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

  std::tuple<Value, Value>
  allocateBufferManuallyAlign(ConversionPatternRewriter &rewriter, Location loc,
                              Value sizeBytes, OpType op,
                              Value alignment) const;

  /// Allocates a memory buffer using an aligned allocation method.
  Value allocateBufferAutoAlign(ConversionPatternRewriter &rewriter,
                                Location loc, Value sizeBytes, OpType op,
                                int64_t alignment) const;

  virtual std::tuple<Value, Value>
  allocateBuffer(ConversionPatternRewriter &rewriter, Location loc, Value size,
                 OpType op) const = 0;

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
        createIndexAttrConstant(rewriter, loc, indexType, *alignmentAttr);
  } else if (!memRefType.getElementType().isSignlessIntOrIndexOrFloat()) {
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
std::tuple<Value, Value>
AllocationLowering<OpType>::allocateBufferManuallyAlign(
    ConversionPatternRewriter &rewriter, Location loc, Value sizeBytes,
    OpType op, Value alignment) const {
  if (alignment) {
    // Adjust the allocation size to consider alignment.
    sizeBytes = rewriter.create<gccjit::BinaryOp>(
        loc, sizeBytes.getType(), BOp::Plus, sizeBytes, alignment);
  }

  MemRefType memRefType = getMemRefResultType(op);
  // Allocate the underlying buffer.
  Type elementPtrType = this->getElementPtrType(memRefType);
  Value allocatedPtr = rewriter.create<gccjit::CallOp>(
      loc, this->getVoidPtrType(),
      SymbolRefAttr::get(this->getContext(), "malloc"), ValueRange{sizeBytes},
      /* tailcall */ nullptr, /* builtin */ rewriter.getUnitAttr());

  if (!allocatedPtr)
    return std::make_tuple(Value(), Value());
  Value alignedPtr = allocatedPtr;
  if (alignment) {
    // Compute the aligned pointer.
    Value allocatedInt = rewriter.create<gccjit::BitCastOp>(
        loc, this->getIndexType(), allocatedPtr);
    Value alignmentInt = createAligned(rewriter, loc, allocatedInt, alignment);
    alignedPtr =
        rewriter.create<gccjit::BitCastOp>(loc, elementPtrType, alignmentInt);
  } else {
    alignedPtr =
        rewriter.create<gccjit::BitCastOp>(loc, elementPtrType, allocatedPtr);
  }

  return std::make_tuple(allocatedPtr, alignedPtr);
}

template <typename OpType>
Value AllocationLowering<OpType>::allocateBufferAutoAlign(
    ConversionPatternRewriter &rewriter, Location loc, Value sizeBytes,
    OpType op, int64_t alignment) const {
  Value allocAlignment =
      createIndexAttrConstant(rewriter, loc, this->getIndexType(), alignment);

  MemRefType memRefType = getMemRefResultType(op);
  sizeBytes = createAligned(rewriter, loc, sizeBytes, allocAlignment);

  Type elementPtrType = this->getElementPtrType(memRefType);
  auto result = rewriter.create<gccjit::CallOp>(
      loc, this->getVoidPtrType(),
      SymbolRefAttr::get(this->getContext(), "aligned_alloc"),
      ValueRange{allocAlignment, sizeBytes},
      /* tailcall */ nullptr, /* builtin */ rewriter.getUnitAttr());

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
  auto exprBundle = rewriter.replaceOpWithNewOp<ExprOp>(op, convertedType);
  auto *block = rewriter.createBlock(&exprBundle.getBody());
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);
    // Get actual sizes of the memref as values: static sizes are constant
    // values and dynamic sizes are passed to 'alloc' as operands.  In case of
    // zero-dimensional memref, assume a scalar (size 1).
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value size;

    this->getMemRefDescriptorSizes(loc, memRefType, adaptor.getOperands(),
                                   rewriter, sizes, strides, size, true);

    // Allocate the underlying buffer.
    auto [allocatedPtr, alignedPtr] =
        this->allocateBuffer(rewriter, loc, size, op);

    if (!allocatedPtr || !alignedPtr)
      return rewriter.notifyMatchFailure(loc,
                                         "underlying buffer allocation failed");

    auto arrayTy = ArrayType::get(rewriter.getContext(), this->getIndexType(),
                                  memRefType.getRank());
    auto sizeArr = rewriter.create<gccjit::NewArrayOp>(loc, arrayTy, sizes);
    auto strideArr = rewriter.create<gccjit::NewArrayOp>(loc, arrayTy, strides);
    auto zero =
        this->createIndexAttrConstant(rewriter, loc, this->getIndexType(), 0);
    // Create the MemRef descriptor.
    auto memRefDescriptor = rewriter.create<gccjit::NewStructOp>(
        loc, convertedType, ArrayRef<int32_t>{0, 1, 2, 3, 4},
        ValueRange{alignedPtr, allocatedPtr, zero, sizeArr, strideArr});

    // Return the final value of the descriptor.
    rewriter.create<ReturnOp>(loc, memRefDescriptor);
  }
  return success();
}

struct AllocaOpLowering : public AllocationLowering<memref::AllocaOp> {
  using AllocationLowering<memref::AllocaOp>::AllocationLowering;
  std::tuple<Value, Value>
  allocateBuffer(ConversionPatternRewriter &rewriter, Location loc, Value size,
                 memref::AllocaOp op) const override final {
    auto allocaOp = cast<memref::AllocaOp>(op);
    auto elementType =
        typeConverter->convertType(allocaOp.getType().getElementType());

    if (allocaOp.getType().getMemorySpace())
      return std::make_tuple(Value(), Value());

    auto elementPtrType = PointerType::get(rewriter.getContext(), elementType);

    Value alloca;

    if (auto align = op.getAlignment()) {
      auto alignment =
          createIndexAttrConstant(rewriter, loc, getIndexType(), *align);
      alloca = rewriter
                   .create<CallOp>(loc, getVoidPtrType(),
                                   SymbolRefAttr::get(rewriter.getContext(),
                                                      "alloca_with_align"),
                                   ValueRange{size, alignment},
                                   /* tailcall */ nullptr,
                                   /* builtin */ rewriter.getUnitAttr())
                   .getResult();
    } else {
      alloca = rewriter
                   .create<CallOp>(
                       loc, getVoidPtrType(),
                       SymbolRefAttr::get(rewriter.getContext(), "alloca"),
                       ValueRange{size},
                       /* tailcall */ nullptr,
                       /* builtin */ rewriter.getUnitAttr())
                   .getResult();
    }

    alloca = rewriter.create<BitCastOp>(loc, elementPtrType, alloca);

    return std::make_tuple(alloca, alloca);
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
  patterns.insert<LoadOpLowering, StoreOpLowering, AllocaOpLowering>(
      typeConverter, &getContext());
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
