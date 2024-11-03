// Copyright 2024 Sirui Mu
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

#ifndef MLIR_GCCJIT_IR_GCCJIT_TYPES_H
#define MLIR_GCCJIT_IR_GCCJIT_TYPES_H

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <libgccjit.h>

#include "mlir-gccjit/IR/GCCJITAttrs.h"
#include "mlir-gccjit/IR/Interfaces/GCCJITRecordTypeInterface.h"

namespace mlir::gccjit {
class FieldAttr;
class SourceLocAttr;
} // namespace mlir::gccjit

#define GET_TYPEDEF_CLASSES
#include "mlir-gccjit/IR/GCCJITOpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// GCCJIT Custom Parser/Printer Signatures
//===----------------------------------------------------------------------===//
namespace mlir::gccjit {
mlir::ParseResult parseFuncTypeArgs(mlir::AsmParser &p,
                                    llvm::SmallVector<mlir::Type> &params,
                                    bool &isVarArg);
void printFuncTypeArgs(mlir::AsmPrinter &p, mlir::ArrayRef<mlir::Type> params,
                       bool isVarArg);

inline bool isIntegral(mlir::Type type) {
  if (auto qualified = dyn_cast<QualifiedType>(type))
    return isIntegral(qualified.getElementType());

  return isa<IntType>(type);
}

inline bool isIntegralOrPointer(mlir::Type type) {
  if (auto qualified = dyn_cast<QualifiedType>(type))
    return isIntegralOrPointer(qualified.getElementType());
  return isa<IntType>(type) || isa<PointerType>(type);
}

inline bool isPointer(mlir::Type type) {
  if (auto qualified = dyn_cast<QualifiedType>(type))
    return isPointer(qualified.getElementType());
  return isa<PointerType>(type);
}

inline bool isArithmetc(mlir::Type type) {
  if (auto qualified = dyn_cast<QualifiedType>(type))
    return isArithmetc(qualified.getElementType());
  return isa<IntType>(type) || isa<FloatType>(type);
}

inline bool isArithmetcOrPointer(mlir::Type type) {
  if (auto qualified = dyn_cast<QualifiedType>(type))
    return isArithmetcOrPointer(qualified.getElementType());
  return isa<IntType>(type) || isa<FloatType>(type) || isa<PointerType>(type);
}
} // namespace mlir::gccjit

#endif // MLIR_GCCJIT_IR_GCCJIT_TYPES_H
