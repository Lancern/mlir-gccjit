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

#include <libgccjit.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

#include "mlir-gccjit/IR/GCCJITAttrs.h"
#include "mlir-gccjit/IR/GCCJITTypes.h"
#include "mlir/IR/MLIRContext.h"
#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;
using namespace mlir::gccjit;

GCCJITTypeConverter::GCCJITTypeConverter() : TypeConverter() {
  addConversion([&](mlir::IndexType type) { return convertIndexType(type); });
  addConversion(
      [&](mlir::IntegerType type) { return convertIntegerType(type); });
  addConversion([&](mlir::FloatType type) { return convertFloatType(type); });
  addConversion(
      [&](mlir::ComplexType type) { return convertComplexType(type); });
  addConversion([&](mlir::VectorType type) { return convertVectorType(type); });
  addConversion([&](mlir::FunctionType type) {
    return convertFunctionTypeAsPtr(type, false);
  });
  addConversion(
      [&](mlir::MemRefType type) { return getMemrefDescriptorType(type); });
}

// Nothing to do for now
GCCJITTypeConverter::~GCCJITTypeConverter() {}

gccjit::IntType
GCCJITTypeConverter::convertIndexType(mlir::IndexType type) const {
  return IntType::get(type.getContext(), GCC_JIT_TYPE_SIZE_T);
}

gccjit::IntType
GCCJITTypeConverter::convertIntegerType(mlir::IntegerType type) const {
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
GCCJITTypeConverter::convertIntegerAttr(mlir::IntegerAttr attr) const {
  auto value = attr.getValue();
  if (auto intType = dyn_cast<IntegerType>(attr.getType())) {
    auto type = convertIntegerType(intType);
    return IntAttr::get(attr.getContext(), type, value);
  }

  if (auto indexType = dyn_cast<IndexType>(attr.getType())) {
    auto type = convertIndexType(indexType);
    return IntAttr::get(attr.getContext(), type, value);
  }

  return {};
}

gccjit::FloatType
GCCJITTypeConverter::convertFloatType(mlir::FloatType type) const {
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

gccjit::FloatAttr
GCCJITTypeConverter::convertFloatAttr(mlir::FloatAttr attr) const {
  auto value = attr.getValue();
  auto type = convertFloatType(cast<mlir::FloatType>(attr.getType()));
  return FloatAttr::get(attr.getContext(), type, value);
}

gccjit::ComplexType
GCCJITTypeConverter::convertComplexType(mlir::ComplexType type) const {
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
GCCJITTypeConverter::convertVectorType(mlir::VectorType type) const {
  auto elementType = convertType(type.getElementType());
  auto size = type.getNumElements();
  return VectorType::get(type.getContext(), elementType, size);
}

gccjit::FuncType
GCCJITTypeConverter::convertFunctionType(mlir::FunctionType type,
                                         bool isVarArg) const {
  llvm::SmallVector<Type> argTypes;
  argTypes.reserve(type.getNumInputs());
  if (convertTypes(type.getInputs(), argTypes).failed())
    return {};
  auto resultType =
      convertAndPackTypesIfNonSingleton(type.getResults(), type.getContext());
  return FuncType::get(type.getContext(), argTypes, resultType, isVarArg);
}

gccjit::PointerType
GCCJITTypeConverter::convertFunctionTypeAsPtr(mlir::FunctionType type,
                                              bool isVarArg) const {
  auto funcType = convertFunctionType(type, isVarArg);
  return PointerType::get(type.getContext(), funcType);
}

gccjit::StructType
GCCJITTypeConverter::getMemrefDescriptorType(mlir::MemRefType type) const {
  std::string name;
  llvm::raw_string_ostream os(name);
  type.print(os);
  os.flush();
  auto nameAttr = StringAttr::get(type.getContext(), name);
  auto elementType = convertType(type.getElementType());
  auto elementPtrType = PointerType::get(type.getContext(), elementType);
  auto indexType = IntType::get(type.getContext(), GCC_JIT_TYPE_SIZE_T);
  auto rank = type.getRank();
  auto dimOrStrideType =
      gccjit::ArrayType::get(type.getContext(), indexType, rank);
  SmallVector<Attribute> fields;
  llvm::StringRef names[]{
      "base", "aligned", "offset", "sizes", "strides",
  };
  for (auto [idx, field] :
       llvm::enumerate(ArrayRef<Type>{elementPtrType, elementPtrType, indexType,
                                      dimOrStrideType, dimOrStrideType})) {
    auto nameAttr = StringAttr::get(type.getContext(), names[idx]);
    fields.push_back(FieldAttr::get(type.getContext(), nameAttr, field));
  }
  auto fieldsAttr = ArrayAttr::get(type.getContext(), fields);
  return StructType::get(type.getContext(), nameAttr, fieldsAttr);
}

gccjit::StructType GCCJITTypeConverter::getUnrankedMemrefDescriptorType(
    mlir::UnrankedMemRefType type) const {

  auto name =
      Twine("__unranked_memref_")
          .concat(Twine(reinterpret_cast<uintptr_t>(type.getAsOpaquePointer())))
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
    fields.push_back(FieldAttr::get(type.getContext(), nameAttr, field));
  }
  auto fieldsAttr = ArrayAttr::get(type.getContext(), fields);
  return StructType::get(type.getContext(), nameAttr, fieldsAttr);
}

Type GCCJITTypeConverter::convertAndPackTypesIfNonSingleton(
    TypeRange types, MLIRContext *ctx) const {
  if (types.size() == 0)
    return VoidType::get(ctx);
  if (types.size() == 1)
    return convertType(types.front());

  auto *name = "__return_pack";
  SmallVector<Attribute> fields;
  for (auto [idx, type] : llvm::enumerate(types)) {
    auto name = Twine("__field_").concat(Twine(idx)).str();
    auto nameAttr = StringAttr::get(ctx, name);
    fields.push_back(FieldAttr::get(type.getContext(), nameAttr, type));
  }
  auto nameAttr = StringAttr::get(ctx, name);
  auto fieldsAttr = ArrayAttr::get(ctx, fields);
  return StructType::get(ctx, nameAttr, fieldsAttr);
}

bool GCCJITTypeConverter::isSigned(gccjit::IntType type) const {
  switch (type.getKind()) {
  case GCC_JIT_TYPE_UNSIGNED_INT:
  case GCC_JIT_TYPE_UNSIGNED_LONG:
  case GCC_JIT_TYPE_UNSIGNED_LONG_LONG:
  case GCC_JIT_TYPE_UINT8_T:
  case GCC_JIT_TYPE_UINT16_T:
  case GCC_JIT_TYPE_UINT32_T:
  case GCC_JIT_TYPE_UINT64_T:
  case GCC_JIT_TYPE_UINT128_T:
    return false;
  default:
    return true;
  }
}

gccjit::IntType GCCJITTypeConverter::makeSigned(gccjit::IntType type) const {
  switch (type.getKind()) {
  case GCC_JIT_TYPE_UNSIGNED_INT:
    return IntType::get(type.getContext(), GCC_JIT_TYPE_INT);
  case GCC_JIT_TYPE_UNSIGNED_LONG:
    return IntType::get(type.getContext(), GCC_JIT_TYPE_LONG);
  case GCC_JIT_TYPE_UNSIGNED_LONG_LONG:
    return IntType::get(type.getContext(), GCC_JIT_TYPE_LONG_LONG);
  case GCC_JIT_TYPE_UINT8_T:
    return IntType::get(type.getContext(), GCC_JIT_TYPE_INT8_T);
  case GCC_JIT_TYPE_UINT16_T:
    return IntType::get(type.getContext(), GCC_JIT_TYPE_INT16_T);
  case GCC_JIT_TYPE_UINT32_T:
    return IntType::get(type.getContext(), GCC_JIT_TYPE_INT32_T);
  case GCC_JIT_TYPE_UINT64_T:
    return IntType::get(type.getContext(), GCC_JIT_TYPE_INT64_T);
  case GCC_JIT_TYPE_UINT128_T:
    return IntType::get(type.getContext(), GCC_JIT_TYPE_INT128_T);
  default:
    return type;
  }
}

// the counterpart of makeSigned
gccjit::IntType GCCJITTypeConverter::makeUnsigned(gccjit::IntType type) const {
  switch (type.getKind()) {
  case GCC_JIT_TYPE_INT:
    return IntType::get(type.getContext(), GCC_JIT_TYPE_UNSIGNED_INT);
  case GCC_JIT_TYPE_LONG:
    return IntType::get(type.getContext(), GCC_JIT_TYPE_UNSIGNED_LONG);
  case GCC_JIT_TYPE_LONG_LONG:
    return IntType::get(type.getContext(), GCC_JIT_TYPE_UNSIGNED_LONG_LONG);
  case GCC_JIT_TYPE_INT8_T:
    return IntType::get(type.getContext(), GCC_JIT_TYPE_UINT8_T);
  case GCC_JIT_TYPE_INT16_T:
    return IntType::get(type.getContext(), GCC_JIT_TYPE_UINT16_T);
  case GCC_JIT_TYPE_INT32_T:
    return IntType::get(type.getContext(), GCC_JIT_TYPE_UINT32_T);
  case GCC_JIT_TYPE_INT64_T:
    return IntType::get(type.getContext(), GCC_JIT_TYPE_UINT64_T);
  case GCC_JIT_TYPE_INT128_T:
    return IntType::get(type.getContext(), GCC_JIT_TYPE_UINT128_T);
  default:
    return type;
  }
}
