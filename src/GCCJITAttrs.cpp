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
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_ATTRDEF_CLASSES
#include "mlir-gccjit/IR/GCCJITOpsAttributes.cpp.inc"

namespace mlir::gccjit {
//===----------------------------------------------------------------------===//
// General GCCJIT parsing / printing
//===----------------------------------------------------------------------===//
Attribute GCCJITDialect::parseAttribute(DialectAsmParser &Parser, Type Type) const {
  llvm::SMLoc TypeLoc = Parser.getCurrentLocation();
  StringRef Mnemonic;
  Attribute GenAttr;
  OptionalParseResult ParseResult = generatedAttributeParser(Parser, &Mnemonic, Type, GenAttr);
  if (ParseResult.has_value())
    return GenAttr;
  Parser.emitError(TypeLoc, "unknown attribute in GCCJIT dialect");
  return Attribute();
}

void GCCJITDialect::printAttribute(Attribute Attr, DialectAsmPrinter &Os) const {
  if (failed(generatedAttributePrinter(Attr, Os)))
    llvm_unreachable("unexpected GCCJIT attribute");
}

//===----------------------------------------------------------------------===//
// TLSModelAttr definitions
//===----------------------------------------------------------------------===//

Attribute TLSModelAttr::parse(AsmParser &Parser, Type OdsType) {
  auto Loc = Parser.getCurrentLocation();
  if (Parser.parseLess())
    return {};

  // Parse variable 'lang'.
  llvm::StringRef Model;
  if (Parser.parseKeyword(&Model))
    return {};

  // Check if parsed value is a valid language.
  auto ModelEnum = symbolizeTLSModelEnum(Model);
  if (!ModelEnum.has_value()) {
    Parser.emitError(Loc) << "invalid TLS model keyword '" << Model << "'";
    return {};
  }

  if (Parser.parseGreater())
    return {};

  return get(Parser.getContext(), TLSModelEnumAttr::get(Parser.getContext(), ModelEnum.value()));
}

void TLSModelAttr::print(AsmPrinter &Printer) const {
  Printer << "<" << getModel().getValue() << '>';
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
