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

#include "mlir-gccjit/Conversion/TypeConverter.h"
#include "libgccjit.h"
#include "mlir-gccjit/IR/GCCJITAttrs.h"
#include "mlir-gccjit/IR/GCCJITTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::gccjit;

GCCJITTypeConverter::GCCJITTypeConverter() : TypeConverter(), packedTypes() {
  addConversion([&](mlir::IndexType type) { return convertIndexType(type); });
  addConversion(
      [&](mlir::IntegerType type) { return convertIntegerType(type); });
  addConversion([&](mlir::FloatType type) { return convertFloatType(type); });
  addConversion(
      [&](mlir::ComplexType type) { return convertComplexType(type); });
  addConversion([&](mlir::VectorType type) { return convertVectorType(type); });
  addConversion([&](mlir::FunctionType type) {
    return convertFunctionType(type, false);
  });
  addConversion(
      [&](mlir::MemRefType type) { return getMemrefDescriptorType(type); });
}

// Nothing to do for now
GCCJITTypeConverter::~GCCJITTypeConverter() {}

gccjit::IntType GCCJITTypeConverter::convertIndexType(mlir::IndexType type) {
  return IntType::get(type.getContext(), GCC_JIT_TYPE_SIZE_T);
}

gccjit::IntType
GCCJITTypeConverter::convertIntegerType(mlir::IntegerType type) {
  // gccjit always translates bitwidth to specific types
  // https://github.com/gcc-mirror/gcc/blob/ae0dbea896b77686fcd1c890e5b7c5fed6197767/gcc/jit/jit-recording.cc#L796
  switch (type.getWidth()) {
  case 1:
    return IntType::get(type.getContext(), GCC_JIT_TYPE_BOOL);
  case 8:
    return IntType::get(type.getContext(), type.isSigned()
                                               ? GCC_JIT_TYPE_INT8_T
                                               : GCC_JIT_TYPE_UINT8_T);
  case 16:
    return IntType::get(type.getContext(), type.isSigned()
                                               ? GCC_JIT_TYPE_INT16_T
                                               : GCC_JIT_TYPE_UINT16_T);
  case 32:
    return IntType::get(type.getContext(), type.isSigned()
                                               ? GCC_JIT_TYPE_INT32_T
                                               : GCC_JIT_TYPE_UINT32_T);
  case 64:
    return IntType::get(type.getContext(), type.isSigned()
                                               ? GCC_JIT_TYPE_INT64_T
                                               : GCC_JIT_TYPE_UINT64_T);
  case 128:
    return IntType::get(type.getContext(), type.isSigned()
                                               ? GCC_JIT_TYPE_INT128_T
                                               : GCC_JIT_TYPE_UINT128_T);
  default:
    return {};
  }
}

gccjit::IntAttr
GCCJITTypeConverter::convertIntegerAttr(mlir::IntegerAttr attr) {
  auto value = attr.getValue();
  auto type = convertIntegerType(cast<IntegerType>(attr.getType()));
  return IntAttr::get(attr.getContext(), type, value);
}

gccjit::FloatType GCCJITTypeConverter::convertFloatType(mlir::FloatType type) {
  if (type.isF32())
    return FloatType::get(type.getContext(), GCC_JIT_TYPE_FLOAT);
  if (type.isF64())
    return FloatType::get(type.getContext(), GCC_JIT_TYPE_DOUBLE);

  // FIXME: we cannot really distinguish between f80 and f128 for GCCJIT, maybe
  // we need target information.
  if (type.isF80() || type.isF128())
    return FloatType::get(type.getContext(), GCC_JIT_TYPE_LONG_DOUBLE);

  return {};
}

gccjit::FloatAttr GCCJITTypeConverter::convertFloatAttr(mlir::FloatAttr attr) {
  auto value = attr.getValue();
  auto type = convertFloatType(cast<mlir::FloatType>(attr.getType()));
  return FloatAttr::get(attr.getContext(), type, value);
}

gccjit::ComplexType
GCCJITTypeConverter::convertComplexType(mlir::ComplexType type) {
  auto elementType = type.getElementType();
  if (elementType.isF32())
    return ComplexType::get(type.getContext(), GCC_JIT_TYPE_COMPLEX_FLOAT);
  if (elementType.isF64())
    return ComplexType::get(type.getContext(), GCC_JIT_TYPE_COMPLEX_DOUBLE);
  if (elementType.isF80() || elementType.isF128())
    return ComplexType::get(type.getContext(),
                            GCC_JIT_TYPE_COMPLEX_LONG_DOUBLE);
  return {};
}

gccjit::VectorType
GCCJITTypeConverter::convertVectorType(mlir::VectorType type) {
  auto elementType = convertType(type.getElementType());
  auto size = type.getNumElements();
  return VectorType::get(type.getContext(), elementType, size);
}

gccjit::FuncType
GCCJITTypeConverter::convertFunctionType(mlir::FunctionType type,
                                         bool isVarArg) {
  llvm::SmallVector<Type> argTypes;
  argTypes.reserve(type.getNumInputs());
  if (convertTypes(type.getInputs(), argTypes).failed())
    return {};
  auto resultType = convertAndPackTypesIfNonSingleton(type.getResults(), type);
  return FuncType::get(type.getContext(), argTypes, resultType, isVarArg);
}

gccjit::PointerType
GCCJITTypeConverter::convertFunctionTypeAsPtr(mlir::FunctionType type,
                                              bool isVarArg) {
  auto funcType = convertFunctionType(type, isVarArg);
  return PointerType::get(type.getContext(), funcType);
}

gccjit::StructType
GCCJITTypeConverter::getMemrefDescriptorType(mlir::MemRefType type) {
  auto &cached = packedTypes[type];
  if (!cached) {
    auto name = Twine("__memref_")
                    .concat(Twine(
                        reinterpret_cast<uintptr_t>(type.getAsOpaquePointer())))
                    .str();
    auto nameAttr = StringAttr::get(type.getContext(), name);
    auto elementType = convertType(type.getElementType());
    auto elementPtrType = PointerType::get(type.getContext(), elementType);
    auto indexType = IntType::get(type.getContext(), GCC_JIT_TYPE_SIZE_T);
    auto rank = type.getRank();
    auto dimOrStrideType =
        gccjit::ArrayType::get(type.getContext(), indexType, rank);
    SmallVector<Attribute> fields;
    for (auto [idx, field] : llvm::enumerate(
             ArrayRef<Type>{elementPtrType, elementPtrType, indexType,
                            dimOrStrideType, dimOrStrideType})) {
      auto name = Twine("__field_").concat(Twine(idx)).str();
      auto nameAttr = StringAttr::get(type.getContext(), name);
      fields.push_back(FieldAttr::get(type.getContext(), nameAttr, field, 0));
    }
    auto fieldsAttr = ArrayAttr::get(type.getContext(), fields);
    cached = StructType::get(type.getContext(), nameAttr, fieldsAttr);
  }
  return cached;
}

gccjit::StructType GCCJITTypeConverter::getUnrankedMemrefDescriptorType(
    mlir::UnrankedMemRefType type) {
  auto &cached = packedTypes[type];
  if (!cached) {
    auto name = Twine("__unranked_memref_")
                    .concat(Twine(
                        reinterpret_cast<uintptr_t>(type.getAsOpaquePointer())))
                    .str();
    auto nameAttr = StringAttr::get(type.getContext(), name);
    auto indexType = IntType::get(type.getContext(), GCC_JIT_TYPE_SIZE_T);
    auto opaquePtrType = PointerType::get(
        type.getContext(), IntType::get(type.getContext(), GCC_JIT_TYPE_VOID));
    SmallVector<Attribute> fields;
    for (auto [idx, field] :
         llvm::enumerate(ArrayRef<Type>{indexType, opaquePtrType})) {
      auto name = Twine("__field_").concat(Twine(idx)).str();
      auto nameAttr = StringAttr::get(type.getContext(), name);
      fields.push_back(FieldAttr::get(type.getContext(), nameAttr, field, 0));
    }
    auto fieldsAttr = ArrayAttr::get(type.getContext(), fields);
    cached = StructType::get(type.getContext(), nameAttr, fieldsAttr);
  }
  return cached;
}

Type GCCJITTypeConverter::convertAndPackTypesIfNonSingleton(TypeRange types,
                                                            FunctionType func) {
  if (types.size() == 0)
    return VoidType::get(func.getContext());
  if (types.size() == 1)
    return types.front();
  gccjit::StructType &cached = packedTypes[func];
  if (!cached) {
    auto name = Twine("__retpack_")
                    .concat(Twine(
                        reinterpret_cast<uintptr_t>(func.getAsOpaquePointer())))
                    .str();
    SmallVector<Attribute> fields;
    for (auto [idx, type] : llvm::enumerate(types)) {
      auto name = Twine("__field_").concat(Twine(idx)).str();
      auto nameAttr = StringAttr::get(func.getContext(), name);
      fields.push_back(FieldAttr::get(type.getContext(), nameAttr, type, 0));
    }
    auto nameAttr = StringAttr::get(func.getContext(), name);
    auto fieldsAttr = ArrayAttr::get(func.getContext(), fields);
    auto structType = StructType::get(func.getContext(), nameAttr, fieldsAttr);
    cached = structType;
  }
  return cached;
}
