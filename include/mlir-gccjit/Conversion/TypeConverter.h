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

#ifndef MLIR_GCCJIT_CONVERSION_TYPECONVERTER_H
#define MLIR_GCCJIT_CONVERSION_TYPECONVERTER_H

#include <libgccjit.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/Transforms/DialectConversion.h>

#include "mlir-gccjit/IR/GCCJITAttrs.h"
#include "mlir-gccjit/IR/GCCJITTypes.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::gccjit {
class GCCJITTypeConverter : public TypeConverter {

public:
  GCCJITTypeConverter();
  ~GCCJITTypeConverter();
  // integral types
  gccjit::IntType convertIndexType(mlir::IndexType type) const;
  gccjit::IntType makeSigned(gccjit::IntType type) const;
  gccjit::IntType makeUnsigned(gccjit::IntType type) const;
  bool isSigned(gccjit::IntType type) const;
  gccjit::IntType convertIntegerType(mlir::IntegerType type) const;
  gccjit::IntAttr convertIntegerAttr(mlir::IntegerAttr attr) const;

  // floating point types
  gccjit::FloatType convertFloatType(mlir::FloatType type) const;
  gccjit::FloatAttr convertFloatAttr(mlir::FloatAttr attr) const;

  // special composite types
  gccjit::ComplexType convertComplexType(mlir::ComplexType type) const;
  gccjit::VectorType convertVectorType(mlir::VectorType type) const;

  // function prototype
  gccjit::FuncType convertFunctionType(mlir::FunctionType type,
                                       bool isVarArg) const;

  // function type to function pointer
  gccjit::PointerType convertFunctionTypeAsPtr(mlir::FunctionType type,
                                               bool isVarArg) const;

  // memref type
  gccjit::StructType getMemrefDescriptorType(mlir::MemRefType type) const;
  gccjit::StructType
  getUnrankedMemrefDescriptorType(mlir::UnrankedMemRefType type) const;

  Type convertAndPackTypesIfNonSingleton(TypeRange types, MLIRContext *) const;
};
} // namespace mlir::gccjit
#endif // MLIR_GCCJIT_CONVERSION_TYPECONVERTER_H
