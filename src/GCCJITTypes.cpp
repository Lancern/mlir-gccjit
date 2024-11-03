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

#include <optional>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <libgccjit.h>

#include "mlir-gccjit/IR/GCCJITAttrs.h"
#include "mlir-gccjit/IR/GCCJITDialect.h"
#include "mlir-gccjit/IR/GCCJITTypes.h"

using namespace mlir;
using namespace mlir::gccjit;

//===----------------------------------------------------------------------===//
// GCCJIT Custom Parser/Printer Signatures
//===----------------------------------------------------------------------===//

static LogicalResult parseRecordBody(AsmParser &parser, StringAttr &name,
                                     ArrayAttr &fields,
                                     std::optional<SourceLocAttr> &loc) {
  if (parser.parseAttribute(name))
    return failure();

  if (parser.parseLBrace())
    return failure();

  SmallVector<Attribute> fieldAttrs;
  auto fieldParser = [&] {
    FieldAttr attr;
    if (failed(parser.parseAttribute(attr)))
      return failure();
    fieldAttrs.push_back(attr);
    return success();
  };
  if (parser.parseCommaSeparatedList(fieldParser))
    return failure();
  fields = ArrayAttr::get(parser.getContext(), fieldAttrs);

  if (parser.parseRBrace())
    return failure();

  SourceLocAttr locAttr;
  OptionalParseResult parseLocResult = parser.parseOptionalAttribute(locAttr);
  if (parseLocResult.has_value() && parseLocResult.value())
    return failure();
  if (locAttr)
    loc.emplace(locAttr);
  else
    loc.reset();

  return success();
}

static void printRecordBody(AsmPrinter &printer, StringAttr name,
                            ArrayAttr fields,
                            std::optional<SourceLocAttr> loc) {
  printer << name << " {";
  llvm::interleaveComma(fields, printer, [&printer](mlir::Attribute elem) {
    printer << cast<FieldAttr>(elem);
  });
  printer << "}";
  if (loc)
    printer << " " << *loc;
}

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
Type GCCJITDialect::parseType(DialectAsmParser &parser) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  StringRef mnemonic;
  Type genType;

  // Try to parse as a tablegen'd type.
  OptionalParseResult parseResult =
      generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  // TODO: add this for custom types
  // Type is not tablegen'd: try to parse as a raw C++ type.
  return StringSwitch<function_ref<Type()>>(mnemonic).Default([&] {
    parser.emitError(typeLoc) << "unknown GCCJIT type: " << mnemonic;
    return Type();
  })();
}

void GCCJITDialect::printType(Type type, DialectAsmPrinter &os) const {
  // Try to print as a tablegen'd type.
  if (generatedTypePrinter(type, os).succeeded())
    return;
  // TODO: add this for custom types
  // Type is not tablegen'd: try printing as a raw C++ type.
  TypeSwitch<Type>(type).Default([](Type) {
    llvm::report_fatal_error("printer is missing a handler for this type");
  });
}
} // namespace mlir::gccjit

//===----------------------------------------------------------------------===//
// FuncType Definitions
//===----------------------------------------------------------------------===//

mlir::ParseResult mlir::gccjit::parseFuncTypeArgs(
    mlir::AsmParser &p, llvm::SmallVector<mlir::Type> &params, bool &isVarArg) {
  isVarArg = false;
  // `(` `)`
  if (succeeded(p.parseOptionalRParen()))
    return mlir::success();

  // `(` `...` `)`
  if (succeeded(p.parseOptionalEllipsis())) {
    isVarArg = true;
    return p.parseRParen();
  }

  // type (`,` type)* (`,` `...`)?
  mlir::Type type;
  if (p.parseType(type))
    return mlir::failure();
  params.push_back(type);
  while (succeeded(p.parseOptionalComma())) {
    if (succeeded(p.parseOptionalEllipsis())) {
      isVarArg = true;
      return p.parseRParen();
    }
    if (p.parseType(type))
      return mlir::failure();
    params.push_back(type);
  }

  return p.parseRParen();
}

void mlir::gccjit::printFuncTypeArgs(mlir::AsmPrinter &p,
                                     mlir::ArrayRef<mlir::Type> params,
                                     bool isVarArg) {
  llvm::interleaveComma(params, p,
                        [&p](mlir::Type type) { p.printType(type); });
  if (isVarArg) {
    if (!params.empty())
      p << ", ";
    p << "...";
  }
  p << ')';
}

llvm::ArrayRef<mlir::Type> FuncType::getReturnTypes() const {
  return static_cast<detail::FuncTypeStorage *>(getImpl())->returnType;
}

bool FuncType::isVoid() const { return mlir::isa<VoidType>(getReturnType()); }

FuncType FuncType::clone(TypeRange inputs, TypeRange results) const {
  assert(results.size() == 1 && "expected exactly one result type");
  return get(llvm::to_vector(inputs), results[0], isVarArg());
}

//===----------------------------------------------------------------------===//
// QualifiedType Definitions
//===----------------------------------------------------------------------===//

mlir::Type mlir::gccjit::QualifiedType::parse(::mlir::AsmParser &odsParser) {
  if (odsParser.parseLess())
    return {};

  mlir::Type elementType;
  if (odsParser.parseType(elementType))
    return {};

  bool isConst = false;
  bool isRestrict = false;
  bool isVolatile = false;

  while (odsParser.parseOptionalComma().succeeded()) {
    llvm::StringRef qualifier;
    if (odsParser.parseKeyword(&qualifier))
      return {};
    if (qualifier == "const") {
      isConst = true;
    } else if (qualifier == "restrict") {
      isRestrict = true;
    } else if (qualifier == "volatile") {
      isVolatile = true;
    } else {
      odsParser.emitError(odsParser.getCurrentLocation(), "unknown qualifier: ")
          << qualifier;
      return {};
    }
  }

  if (odsParser.parseGreater())
    return {};

  return QualifiedType::get(odsParser.getContext(), elementType, isConst,
                            isRestrict, isVolatile);
}

void mlir::gccjit::QualifiedType::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << "<";
  odsPrinter.printType(getElementType());
  if (getIsConst())
    odsPrinter << ", const";
  if (getIsRestrict())
    odsPrinter << ", restrict";
  if (getIsVolatile())
    odsPrinter << ", volatile";
  odsPrinter << ">";
}

//===----------------------------------------------------------------------===//
// Integer Type Definitions
//===----------------------------------------------------------------------===//
mlir::Type mlir::gccjit::IntType::parse(::mlir::AsmParser &odsParser) {
  if (odsParser.parseLess())
    return {};

  std::string keyword{};
  if (odsParser.parseOptionalKeywordOrString(&keyword))
    return {};

  auto kind = llvm::StringSwitch<std::optional<::gcc_jit_types>>(keyword)
                  .Case("bool", GCC_JIT_TYPE_BOOL)
                  .Case("char", GCC_JIT_TYPE_CHAR)
                  .Case("short", GCC_JIT_TYPE_SHORT)
                  .Case("int", GCC_JIT_TYPE_INT)
                  .Case("long", GCC_JIT_TYPE_LONG)
                  .Case("size_t", GCC_JIT_TYPE_SIZE_T)
                  .Case("uint8_t", GCC_JIT_TYPE_UINT8_T)
                  .Case("uint16_t", GCC_JIT_TYPE_UINT16_T)
                  .Case("uint32_t", GCC_JIT_TYPE_UINT32_T)
                  .Case("uint64_t", GCC_JIT_TYPE_UINT64_T)
                  .Case("uint128_t", GCC_JIT_TYPE_INT128_T)
                  .Case("int8_t", GCC_JIT_TYPE_INT8_T)
                  .Case("int16_t", GCC_JIT_TYPE_INT16_T)
                  .Case("int32_t", GCC_JIT_TYPE_INT32_T)
                  .Case("int64_t", GCC_JIT_TYPE_INT64_T)
                  .Case("int128_t", GCC_JIT_TYPE_INT128_T)
                  .Default(std::nullopt);

  if (keyword == "unsigned") {
    if (odsParser.parseOptionalKeywordOrString(&keyword))
      return {};
    kind = llvm::StringSwitch<std::optional<::gcc_jit_types>>(keyword)
               .Case("char", GCC_JIT_TYPE_UNSIGNED_CHAR)
               .Case("short", GCC_JIT_TYPE_UNSIGNED_SHORT)
               .Case("int", GCC_JIT_TYPE_UNSIGNED_INT)
               .Case("long", GCC_JIT_TYPE_UNSIGNED_LONG)
               .Default(std::nullopt);
  }

  if (kind == GCC_JIT_TYPE_LONG || kind == GCC_JIT_TYPE_UNSIGNED_LONG)
    if (odsParser.parseOptionalKeyword("long").succeeded())
      kind = kind == GCC_JIT_TYPE_LONG ? GCC_JIT_TYPE_LONG_LONG
                                       : GCC_JIT_TYPE_UNSIGNED_LONG_LONG;

  if (!kind.has_value()) {
    odsParser.emitError(odsParser.getCurrentLocation(),
                        "unknown integer type: ")
        << keyword;
    return {};
  }

  if (odsParser.parseGreater())
    return {};

  return gccjit::IntType::get(odsParser.getContext(), *kind);
}

void mlir::gccjit::IntType::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << "<";
  switch (getKind()) {
  case GCC_JIT_TYPE_BOOL:
    odsPrinter << "bool";
    break;
  case GCC_JIT_TYPE_CHAR:
    odsPrinter << "char";
    break;
  case GCC_JIT_TYPE_SHORT:
    odsPrinter << "short";
    break;
  case GCC_JIT_TYPE_INT:
    odsPrinter << "int";
    break;
  case GCC_JIT_TYPE_LONG:
    odsPrinter << "long";
    break;
  case GCC_JIT_TYPE_SIZE_T:
    odsPrinter << "size_t";
    break;
  case GCC_JIT_TYPE_UINT8_T:
    odsPrinter << "uint8_t";
    break;
  case GCC_JIT_TYPE_UINT16_T:
    odsPrinter << "uint16_t";
    break;
  case GCC_JIT_TYPE_UINT32_T:
    odsPrinter << "uint32_t";
    break;
  case GCC_JIT_TYPE_UINT64_T:
    odsPrinter << "uint64_t";
    break;
  case GCC_JIT_TYPE_UINT128_T:
    odsPrinter << "uint128_t";
    break;
  case GCC_JIT_TYPE_INT8_T:
    odsPrinter << "int8_t";
    break;
  case GCC_JIT_TYPE_INT16_T:
    odsPrinter << "int16_t";
    break;
  case GCC_JIT_TYPE_INT32_T:
    odsPrinter << "int32_t";
    break;
  case GCC_JIT_TYPE_INT64_T:
    odsPrinter << "int64_t";
    break;
  case GCC_JIT_TYPE_INT128_T:
    odsPrinter << "int128_t";
    break;
  case GCC_JIT_TYPE_UNSIGNED_CHAR:
    odsPrinter << "unsigned char";
    break;
  case GCC_JIT_TYPE_UNSIGNED_SHORT:
    odsPrinter << "unsigned short";
    break;
  case GCC_JIT_TYPE_UNSIGNED_INT:
    odsPrinter << "unsigned int";
    break;
  case GCC_JIT_TYPE_UNSIGNED_LONG:
    odsPrinter << "unsigned long";
    break;
  case GCC_JIT_TYPE_UNSIGNED_LONG_LONG:
    odsPrinter << "unsigned long long";
    break;
  default:
    llvm_unreachable("unknown integer type");
  }
  odsPrinter << ">";
}

//===----------------------------------------------------------------------===//
// Float Type Definitions
//===----------------------------------------------------------------------===//
template <class F, class SingleSwitcher>
static mlir::Type parseFloatingPoint(::mlir::AsmParser &odsParser, F &&mapper,
                                     SingleSwitcher &&switcher) {
  if (odsParser.parseLess())
    return {};
  std::string keyword{};
  if (odsParser.parseOptionalKeywordOrString(&keyword))
    return {};
  auto kind = switcher(keyword);
  if (kind.has_value() && *kind == GCC_JIT_TYPE_LONG_DOUBLE)
    if (odsParser.parseKeyword("double")) {
      odsParser.emitError(odsParser.getCurrentLocation(),
                          "expected 'double' after 'long'");
      return {};
    }

  if (!kind.has_value()) {
    odsParser.emitError(odsParser.getCurrentLocation(), "unknown float type: ")
        << keyword;
    return {};
  }

  if (odsParser.parseGreater())
    return {};

  return mapper(odsParser.getContext(), *kind);
}

mlir::Type mlir::gccjit::FloatType::parse(::mlir::AsmParser &odsParser) {
  return parseFloatingPoint(
      odsParser,
      [](mlir::MLIRContext *ctx, ::gcc_jit_types kind) {
        return gccjit::FloatType::get(ctx, kind);
      },
      /* may add in bfloat16, half, etc. */
      [](llvm::StringRef keyword) -> std::optional<::gcc_jit_types> {
        return llvm::StringSwitch<std::optional<::gcc_jit_types>>(keyword)
            .Case("float", GCC_JIT_TYPE_FLOAT)
            .Case("double", GCC_JIT_TYPE_DOUBLE)
            .Case("long", GCC_JIT_TYPE_LONG_DOUBLE)
            .Default(std::nullopt);
      });
}

void mlir::gccjit::FloatType::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << "<";
  switch (getKind()) {
  case GCC_JIT_TYPE_FLOAT:
    odsPrinter << "float";
    break;
  case GCC_JIT_TYPE_DOUBLE:
    odsPrinter << "double";
    break;
  case GCC_JIT_TYPE_LONG_DOUBLE:
    odsPrinter << "long double";
    break;
  default:
    llvm_unreachable("unknown float type");
  }
  odsPrinter << ">";
}

//===----------------------------------------------------------------------===//
// Complex Type Definitions
//===----------------------------------------------------------------------===//

mlir::Type mlir::gccjit::ComplexType::parse(::mlir::AsmParser &odsParser) {
  return parseFloatingPoint(
      odsParser,
      [](mlir::MLIRContext *ctx, ::gcc_jit_types kind) {
        return gccjit::ComplexType::get(ctx, kind);
      },
      [](llvm::StringRef keyword) -> std::optional<::gcc_jit_types> {
        return llvm::StringSwitch<std::optional<::gcc_jit_types>>(keyword)
            .Case("float", GCC_JIT_TYPE_COMPLEX_FLOAT)
            .Case("double", GCC_JIT_TYPE_COMPLEX_DOUBLE)
            .Case("long", GCC_JIT_TYPE_COMPLEX_LONG_DOUBLE)
            .Default(std::nullopt);
      });
}

void mlir::gccjit::ComplexType::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << "<";
  switch (getKind()) {
  case GCC_JIT_TYPE_FLOAT:
    odsPrinter << "float";
    break;
  case GCC_JIT_TYPE_DOUBLE:
    odsPrinter << "double";
    break;
  case GCC_JIT_TYPE_LONG_DOUBLE:
    odsPrinter << "long double";
    break;
  default:
    llvm_unreachable("unknown float type");
  }
  odsPrinter << ">";
}

//===----------------------------------------------------------------------===//
// Struct and union type definitions
//===----------------------------------------------------------------------===//

static LogicalResult
verifyRecordFields(llvm::function_ref<InFlightDiagnostic()> emitError,
                   ArrayAttr fields) {
  for (Attribute elem : fields) {
    if (!isa<FieldAttr>(elem))
      return emitError() << "fields of a record type must be FieldAttr";
  }
  return success();
}

LogicalResult mlir::gccjit::StructType::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, StringAttr name,
    ArrayAttr fields, std::optional<SourceLocAttr> loc) {
  return verifyRecordFields(emitError, fields);
}

llvm::StringRef mlir::gccjit::StructType::getRecordName() const {
  return getName().getValue();
}

mlir::ArrayAttr mlir::gccjit::StructType::getRecordFields() const {
  return getFields();
}

mlir::gccjit::SourceLocAttr mlir::gccjit::StructType::getRecordLoc() const {
  return getLoc().value_or(mlir::gccjit::SourceLocAttr{});
}

LogicalResult mlir::gccjit::UnionType::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, StringAttr name,
    ArrayAttr fields, std::optional<SourceLocAttr> loc) {
  return verifyRecordFields(emitError, fields);
}

llvm::StringRef mlir::gccjit::UnionType::getRecordName() const {
  return getName().getValue();
}

mlir::ArrayAttr mlir::gccjit::UnionType::getRecordFields() const {
  return getFields();
}

mlir::gccjit::SourceLocAttr mlir::gccjit::UnionType::getRecordLoc() const {
  return getLoc().value_or(mlir::gccjit::SourceLocAttr{});
}

bool mlir::gccjit::UnionType::isUnion() const { return true; }

bool mlir::gccjit::StructType::isUnion() const { return false; }
