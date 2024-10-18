// Copyright 2024 Sirui Mu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include "mlir-gccjit/IR/GCCJITDialect.h"
#include "mlir-gccjit/IR/GCCJITTypes.h"

using namespace mlir::gccjit;

#define GET_TYPEDEF_CLASSES
#include "mlir-gccjit/IR/GCCJITOpsTypes.cpp.inc"

void GCCJITDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir-gccjit/IR/GCCJITOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// General GCCJIT parsing / printing
//===----------------------------------------------------------------------===//

namespace mlir::gccjit {
Type GCCJITDialect::parseType(DialectAsmParser &Parser) const {
  llvm::SMLoc TypeLoc = Parser.getCurrentLocation();
  StringRef Mnemonic;
  Type GenType;

  // Try to parse as a tablegen'd type.
  OptionalParseResult ParseResult = generatedTypeParser(Parser, &Mnemonic, GenType);
  if (ParseResult.has_value())
    return GenType;
  // TODO: add this for custom types
  // Type is not tablegen'd: try to parse as a raw C++ type.
  return StringSwitch<function_ref<Type()>>(Mnemonic).Default([&] {
    Parser.emitError(TypeLoc) << "unknown GCCJIT type: " << Mnemonic;
    return Type();
  })();
}

void GCCJITDialect::printType(Type Ty, DialectAsmPrinter &OS) const {
  // Try to print as a tablegen'd type.
  if (generatedTypePrinter(Ty, OS).succeeded())
    return;
  // TODO: add this for custom types
  // Type is not tablegen'd: try printing as a raw C++ type.
  TypeSwitch<Type>(Ty).Default(
      [](Type) { llvm::report_fatal_error("printer is missing a handler for this type"); });
}
} // namespace mlir::gccjit
