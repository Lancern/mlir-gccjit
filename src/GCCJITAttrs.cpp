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

#include "mlir-gccjit/IR/GCCJITAttrs.h"
#include "mlir-gccjit/IR/GCCJITDialect.h"
#include "mlir-gccjit/IR/GCCJITOpsEnums.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <variant>

#include "mlir-gccjit/IR/GCCJITOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir-gccjit/IR/GCCJITOpsAttributes.cpp.inc"

namespace mlir::gccjit {
//===----------------------------------------------------------------------===//
// General GCCJIT parsing / printing
//===----------------------------------------------------------------------===//
Attribute GCCJITDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  StringRef mnemonic;
  Attribute genAttr;
  OptionalParseResult parseResult = generatedAttributeParser(parser, &mnemonic, type, genAttr);
  if (parseResult.has_value())
    return genAttr;
  parser.emitError(typeLoc, "unknown attribute in GCCJIT dialect");
  return Attribute();
}

void GCCJITDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const {
  if (failed(generatedAttributePrinter(attr, os)))
    llvm_unreachable("unexpected GCCJIT attribute");
}

//===----------------------------------------------------------------------===//
// TLSModelAttr definitions
//===----------------------------------------------------------------------===//

Attribute TLSModelAttr::parse(AsmParser &parser, Type odsType) {
  auto loc = parser.getCurrentLocation();
  if (parser.parseLess())
    return {};

  // Parse variable 'lang'.
  llvm::StringRef model;
  if (parser.parseKeyword(&model))
    return {};

  // Check if parsed value is a valid language.
  auto modelEnum = symbolizeTLSModelEnum(model);
  if (!modelEnum.has_value()) {
    parser.emitError(loc) << "invalid TLS model keyword '" << model << "'";
    return {};
  }

  if (parser.parseGreater())
    return {};

  return get(parser.getContext(), TLSModelEnumAttr::get(parser.getContext(), modelEnum.value()));
}

void TLSModelAttr::print(AsmPrinter &printer) const {
  printer << "<" << getModel().getValue() << '>';
}

//===----------------------------------------------------------------------===//
// IntAttr definitions
//===----------------------------------------------------------------------===//

Attribute IntAttr::parse(AsmParser &parser, Type odsType) {
  auto intType = mlir::dyn_cast<IntType>(odsType);
  if (!intType)
    return {};

  // Consume the '<' symbol.
  if (parser.parseLess())
    return {};

  // Fetch arbitrary precision integer value.
  long value;
  if (parser.parseInteger(value))
    parser.emitError(parser.getCurrentLocation(), "expected integer value");
  mlir::APInt storage(sizeof(long) * 8, value, true);
  if (storage.getSExtValue() != value)
    parser.emitError(parser.getCurrentLocation(), "integer value too large for the given type");

  // Consume the '>' symbol.
  if (parser.parseGreater())
    return {};

  return IntAttr::get(parser.getContext(), intType, storage);
}

void IntAttr::print(AsmPrinter &printer) const {
  printer << "<" << getValue() << '>';
}

//===----------------------------------------------------------------------===//
// OptLevelAttr definitions
//===----------------------------------------------------------------------===//
Attribute OptLevelAttr::parse(AsmParser &parser, Type odsType) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseLess())
    return {};

  // Parse variable 'level'.
  llvm::StringRef level;
  if (parser.parseKeyword(&level))
    return {};

  // Check if parsed value is a valid optimization level.
  auto optLevelEnum = symbolizeOptLevelEnum(level);
  if (!optLevelEnum.has_value()) {
    parser.emitError(loc) << "invalid optimization level keyword '" << level << "'";
    return {};
  }

  if (parser.parseGreater())
    return {};

  return get(parser.getContext(), OptLevelEnumAttr::get(parser.getContext(), optLevelEnum.value()));
}

void OptLevelAttr::print(AsmPrinter &printer) const {
  printer << "<" << getLevel().getValue() << '>';
}

//===----------------------------------------------------------------------===//
// GCCJIT Dialect
//===----------------------------------------------------------------------===//

void GCCJITDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir-gccjit/IR/GCCJITOpsAttributes.cpp.inc"
      >();
}

} // namespace mlir::gccjit
