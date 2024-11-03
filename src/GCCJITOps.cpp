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

#include "mlir-gccjit/IR/GCCJITOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BlockSupport.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/DialectInterface.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/StorageUniquerSupport.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include "mlir-gccjit/IR/GCCJITDialect.h"
#include "mlir-gccjit/IR/GCCJITOpsEnums.h"
#include "mlir-gccjit/IR/GCCJITTypes.h"

using namespace mlir;
using namespace mlir::gccjit;

//===----------------------------------------------------------------------===//
// GCCJIT Custom Parser/Printer for Operations
//===----------------------------------------------------------------------===//
namespace {
ParseResult parseFunctionAttrs(OpAsmParser &parser, ArrayAttr &fnAttrs) {
  if (parser.parseOptionalKeyword("attrs").succeeded()) {
    if (parser.parseLParen())
      return failure();
    if (parser.parseAttribute(fnAttrs))
      return failure();
    return parser.parseRParen();
  }
  fnAttrs = ArrayAttr::get(parser.getContext(), {});
  return success();
}

void printFunctionAttrs(OpAsmPrinter &p, Operation *, ArrayAttr fnAttrs) {
  if (fnAttrs && !fnAttrs.empty()) {
    p << "attrs(";
    p.printAttribute(fnAttrs);
    p << ")";
  }
}

ParseResult parseFunctionType(OpAsmParser &parser, TypeAttr &type) {
  Type retType;
  llvm::SmallVector<mlir::Type> params{};
  bool isVarArg = false;
  if (parser.parseLParen())
    return failure();
  if (parseFuncTypeArgs(parser, params, isVarArg))
    return parser.emitError(parser.getCurrentLocation(),
                            "failed to parse function type arguments");
  if (parser.parseOptionalArrow().succeeded()) {
    if (parser.parseType(retType))
      return failure();
  } else {
    retType = gccjit::VoidType::get(parser.getContext());
  }
  gccjit::FuncType funcType = gccjit::FuncType::get(params, retType, isVarArg);
  type = TypeAttr::get(funcType);
  return success();
}

void printFunctionType(OpAsmPrinter &p, Operation *, TypeAttr type) {
  auto funcType = cast<FuncType>(type.getValue());
  p << "(";
  printFuncTypeArgs(p, funcType.getInputs(), funcType.isVarArg());
  if (!isa<gccjit::VoidType>(funcType.getReturnType())) {
    p << " -> ";
    p.printType(funcType.getReturnType());
  }
}

ParseResult parseFunctionBody(OpAsmParser &parser, Region &region) {
  (void)parser.parseOptionalRegion(region);
  return success();
}

void printFunctionBody(OpAsmPrinter &p, Operation *op, Region &region) {
  if (!region.empty())
    p.printRegion(region);
}

/*
      custom<SwitchOpCases>(ref(type($value)),
                              $defaultDestination,
                              $case_lowerbound,
                              $case_upperbound,
                              $caseDestinations)
*/

ParseResult parseSwitchOpCases(OpAsmParser &parser,
                               Type /*todo: check value compatibility*/,
                               Block *&defaultDestinationSuccessor,
                               ArrayAttr &lowerbound, ArrayAttr &upperbound,
                               SmallVectorImpl<Block *> &caseDestinations) {
  llvm::SmallVector<Attribute> lowerboundVec;
  llvm::SmallVector<Attribute> upperboundVec;
  if (parser.parseKeyword("default"))
    return {};
  if (parser.parseArrow())
    return {};
  if (parser.parseSuccessor(defaultDestinationSuccessor))
    return parser.emitError(parser.getCurrentLocation(),
                            "expected default destination successor");
  while (parser.parseOptionalComma().succeeded()) {
    gccjit::IntAttr lowerbound{};
    gccjit::IntAttr upperbound{};
    if (parser.parseCustomAttributeWithFallback<gccjit::IntAttr>(lowerbound))
      return parser.emitError(parser.getCurrentLocation(),
                              "expected lowerbound attribute");
    if (parser.parseOptionalEllipsis().succeeded()) {
      if (parser.parseCustomAttributeWithFallback<gccjit::IntAttr>(upperbound))
        return parser.emitError(parser.getCurrentLocation(),
                                "expected upperbound attribute");
    } else
      upperbound = lowerbound;
    lowerboundVec.push_back(lowerbound);
    upperboundVec.push_back(upperbound);
    if (parser.parseArrow())
      return {};
    Block *caseDestination;
    if (parser.parseSuccessor(caseDestination))
      return parser.emitError(parser.getCurrentLocation(),
                              "expected case destination successor");
    caseDestinations.push_back(caseDestination);
  }
  lowerbound = ArrayAttr::get(parser.getContext(), lowerboundVec);
  upperbound = ArrayAttr::get(parser.getContext(), upperboundVec);
  return success();
};

void printSwitchOpCases(
    OpAsmPrinter &p, Operation *op,
    mlir::gccjit::IntType /*todo: check value compatibility*/,
    Block *defaultDestinationSuccessor, ArrayAttr lowerbound,
    ArrayAttr upperbound, SuccessorRange caseDestinations) {
  p << "default -> ";
  p.printSuccessor(defaultDestinationSuccessor);
  for (auto [lower, upper, dest] :
       llvm::zip(lowerbound, upperbound, caseDestinations)) {
    p << ",";
    p.printNewline();
    p.printAttribute(lower);
    if (lower != upper) {
      p << "...";
      p.printAttribute(upper);
    }
    p << " -> ";
    p.printSuccessor(dest);
  }
  p.printNewline();
}

ParseResult parseGlobalInitializer(OpAsmParser &parser, Attribute &initializer,
                                   Region &body) {
  std::string keyword;
  if (parser.parseOptionalKeywordOrString(&keyword).succeeded()) {
    if (keyword == "literal") {
      if (parser.parseLParen())
        return parser.emitError(parser.getCurrentLocation(),
                                "expected '(' after 'literal'");
      StringAttr stringLiteral;
      if (parser.parseCustomAttributeWithFallback(stringLiteral))
        return parser.emitError(parser.getCurrentLocation(),
                                "expected string initializer");
      initializer = stringLiteral;
      if (parser.parseRParen())
        return parser.emitError(parser.getCurrentLocation(),
                                "expected ')' after string initializer");
      return success();
    }

    if (keyword == "array") {
      if (parser.parseLParen())
        return parser.emitError(parser.getCurrentLocation(),
                                "expected '(' after 'array'");
      ByteArrayInitializerAttr byteArrayInitializer;
      if (parser.parseCustomAttributeWithFallback(byteArrayInitializer))
        return parser.emitError(parser.getCurrentLocation(),
                                "expected byte array initializer");
      initializer = byteArrayInitializer;
      if (parser.parseRParen())
        return parser.emitError(parser.getCurrentLocation(),
                                "expected ')' after byte array initializer");
      return success();
    }

    if (keyword == "init") {
      if (parser.parseRegion(body))
        return parser.emitError(parser.getCurrentLocation(),
                                "expected initializer region");
      return success();
    }

    return parser.emitError(parser.getCurrentLocation(),
                            "unknown initializer kind: ")
           << keyword;
  }
  // There is no initializer.
  return success();
}

void printGlobalInitializer(OpAsmPrinter &p, Operation *op,
                            Attribute initializer, Region &body) {
  if (auto stringLiteral = dyn_cast_if_present<StringAttr>(initializer)) {
    p << "literal(";
    p.printStrippedAttrOrType(stringLiteral);
    p << ")";
    return;
  }

  if (auto byteArrayInitializer =
          dyn_cast_if_present<ByteArrayInitializerAttr>(initializer)) {
    p << "array(";
    p.printStrippedAttrOrType(byteArrayInitializer);
    p << ")";
    return;
  }

  if (!body.empty()) {
    p << "init ";
    p.printRegion(body);
    return;
  }
}

ParseResult parseArrayOrVectorElements(
    OpAsmParser &parser, Type expectedType,
    llvm::SmallVectorImpl<OpAsmParser::UnresolvedOperand> &elementValues,
    llvm::SmallVectorImpl<Type> &elementTypes) {
  llvm_unreachable("Not implemented");
}

void printArrayOrVectorElements(OpAsmPrinter &p, Operation *op,
                                Type expectedType, OperandRange elementValues,
                                ValueTypeRange<OperandRange> elementTypes) {
  llvm_unreachable("Not implemented");
}

struct ParseNamedUnitAttr {
  std::string_view name;
  constexpr ParseNamedUnitAttr(std::string_view name) : name(name) {}
  ParseResult operator()(OpAsmParser &parser, UnitAttr &attr) const {
    if (parser.parseOptionalKeyword(name).succeeded())
      attr = UnitAttr::get(parser.getContext());
    return success();
  }
};

struct PrintNamedUnitAttr {
  std::string_view name;
  constexpr PrintNamedUnitAttr(std::string_view name) : name(name) {}
  void operator()(OpAsmPrinter &p, Operation *, UnitAttr attr) const {
    if (attr)
      p << name;
  }
};

constexpr ParseNamedUnitAttr parseTailCallAttr{"tail"};
constexpr PrintNamedUnitAttr printTailCallAttr{"tail"};
constexpr ParseNamedUnitAttr parseBuiltinCallAttr{"builtin"};
constexpr PrintNamedUnitAttr printBuiltinCallAttr{"builtin"};
constexpr ParseNamedUnitAttr parseAsmInlineAttr{"inline"};
constexpr PrintNamedUnitAttr printAsmInlineAttr{"inline"};
constexpr ParseNamedUnitAttr parseAsmVolatileAttr{"volatile"};
constexpr PrintNamedUnitAttr printAsmVolatileAttr{"volatile"};
constexpr ParseNamedUnitAttr parseLazyAttribute{"lazy"};
constexpr PrintNamedUnitAttr printLazyAttribute{"lazy"};

ParseResult
parseAsmOperands(OpAsmParser &parser, ArrayAttr &constrains, ArrayAttr &symbols,
                 SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
                 SmallVectorImpl<Type> &operandTypes) {
  SmallVector<Attribute> constrainsVec;
  SmallVector<Attribute> symbolsVec;
  do {
    std::string symbol{};
    StringAttr constraint{};
    Type operandType{};
    OpAsmParser::UnresolvedOperand operand;
    // has symbol?
    if (parser.parseOptionalLSquare().succeeded()) {
      if (parser.parseKeywordOrString(&symbol) || symbol.empty())
        return parser.emitError(parser.getCurrentLocation(),
                                "expected  non-empty constraint string");
      if (parser.parseRSquare())
        return {};
    }
    // parse constraint
    if (!parser.parseOptionalAttribute(constraint).has_value()) {
      if (!symbol.empty())
        return parser.emitError(parser.getCurrentLocation(),
                                "expected constraint string after symbol");
      // otherwise, there is no operands. we need to check this for the first
      // round parsing before any comma
      break;
    }

    // parse ( operand )
    if (parser.parseLParen())
      return {};
    if (parser.parseOperand(operand))
      return {};
    if (parser.parseColonType(operandType))
      return {};
    if (parser.parseRParen())
      return {};

    constrainsVec.push_back(constraint);
    symbolsVec.push_back(StringAttr::get(parser.getContext(), symbol));
    operands.push_back(operand);
    operandTypes.push_back(operandType);
  } while (parser.parseOptionalComma().succeeded());

  constrains = ArrayAttr::get(parser.getContext(), constrainsVec);
  symbols = ArrayAttr::get(parser.getContext(), symbolsVec);
  return success();
}
void printAsmOperands(OpAsmPrinter &p, Operation *op, ArrayAttr constrains,
                      ArrayAttr symbols, OperandRange operands,
                      ValueTypeRange<OperandRange> operandTypes) {
  llvm::interleaveComma(llvm::zip(constrains, symbols, operands, operandTypes),
                        p, [&](auto tuple) {
                          auto [constraint, symbol, operand, operandType] =
                              tuple;
                          if (!cast<StringAttr>(symbol).getValue().empty())
                            p << "[\"" << symbol << "\"] ";
                          p.printAttribute(constraint);
                          p << "(";
                          p.printOperand(operand);
                          p << " : ";
                          p.printType(operandType);
                          p << ")";
                        });
}

ParseResult parseClobberList(OpAsmParser &parser, ArrayAttr &clobbers) {
  llvm::SmallVector<Attribute> clobbersVec;
  do {
    StringAttr clobber;
    if (!parser.parseOptionalAttribute(clobber).has_value())
      break;
    clobbersVec.push_back(clobber);
  } while (parser.parseOptionalComma().succeeded());
  clobbers = ArrayAttr::get(parser.getContext(), clobbersVec);
  return success();
}

void printClobberList(OpAsmPrinter &p, Operation *, ArrayAttr clobbers) {
  llvm::interleaveComma(clobbers, p,
                        [&](Attribute clobber) { p.printAttribute(clobber); });
}

} // namespace

#define GET_OP_CLASSES
#include "mlir-gccjit/IR/GCCJITOps.cpp.inc"

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

LogicalResult gccjit::FuncOp::verify() {
  if (getBody().empty() && !isImported())
    return emitOpError("functions with bodies must have at least one block");
  if (isImported()) {
    if (!getBody().empty())
      return emitOpError("external functions cannot have regions");
    return success();
  }
  ValueTypeRange<Region::BlockArgListType> entryArgTys =
      getBody().getArgumentTypes();
  if (entryArgTys.size() != getNumArguments())
    return emitOpError(
        "entry block arguments count should match function arguments count");
  for (auto [protoTy, realTy] : llvm::zip(getArgumentTypes(), entryArgTys)) {
    auto lvalueTy = dyn_cast<LValueType>(realTy);
    if (!lvalueTy)
      return emitOpError("entry block argument should have LValueType");
    if (protoTy != lvalueTy.getInnerType())
      return emitOpError(
          "entry block argument type should match function argument type");
  }

  for (auto attr : getGccjitFnAttrs())
    if (!isa<FunctionAttr>(attr))
      return emitOpError("function attribute should be of FunctionAttr type");

  return success();
}

FlatSymbolRefAttr FuncOp::getAliasee() {
  FlatSymbolRefAttr res{};
  for (auto attr : getGccjitFnAttrs()) {
    auto fnAttr = cast<FunctionAttr>(attr);
    if (fnAttr.getAttr().getValue() == FnAttrEnum::Alias) {
      res = FlatSymbolRefAttr::get(getContext(), fnAttr.getStrValue().value());
      break;
    }
  }
  return res;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto *parent = this->getOperation()->getParentOp();
  if (auto funcOp = dyn_cast<FuncOp>(parent)) {
    auto funcType = funcOp.getFunctionType();
    if (!hasReturnValue() && !funcType.isVoid())
      return emitOpError("must have a return value matching the function type");
    if (hasReturnValue()) {
      if (funcType.isVoid())
        return emitOpError("cannot have a return value for a void function");
      if (funcType.getReturnType() != getValue().getType())
        return emitOpError("return type mismatch");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//
LogicalResult ConstantOp::verify() {
  if (isa<LValueType>(getValue().getType()))
    return emitOpError("value cannot be an lvalue type");
  return success();
}

//===----------------------------------------------------------------------===//
// AsRValueOp
//===----------------------------------------------------------------------===//
LogicalResult AsRValueOp::verify() {
  if (getRvalue().getType() != getLvalue().getType().getInnerType())
    return emitOpError("operand's inner type should match result type");
  return success();
}

//===----------------------------------------------------------------------===//
// RValue Expressions
//===----------------------------------------------------------------------===//
