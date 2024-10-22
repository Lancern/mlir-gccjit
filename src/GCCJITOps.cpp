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
#include "mlir-gccjit/IR/GCCJITDialect.h"

#include "mlir-gccjit/IR/GCCJITOpsEnums.h"
#include "mlir-gccjit/IR/GCCJITTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::gccjit;

//===----------------------------------------------------------------------===//
// GCCJIT Custom Parser/Printer for Operations
//===----------------------------------------------------------------------===//
namespace {

ParseResult parseFunctionKind(OpAsmParser &parser, FnKindAttr &fnKind) {
  std::string kind;
  if (parser.parseOptionalKeywordOrString(&kind))
    return parser.emitError(parser.getNameLoc(), "expected function kind");
  std::optional<FnKind> kindEnum = symbolizeFnKind(kind);
  if (!kindEnum)
    return parser.emitError(parser.getNameLoc(), "unknown function kind: ")
           << kind;
  fnKind = FnKindAttr::get(parser.getContext(), *kindEnum);
  return success();
}

void printFunctionKind(OpAsmPrinter &p, Operation *, FnKindAttr fnKind) {
  p << stringifyFnKind(fnKind.getValue());
}

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

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto funcOp = getParentOp();
  auto funcType = funcOp.getFunctionType();
  if (!hasReturnValue() && !funcType.isVoid())
    return emitOpError("must have a return value matching the function type");
  if (hasReturnValue()) {
    if (funcType.isVoid())
      return emitOpError("cannot have a return value for a void function");
    if (funcType.getReturnType() != getValue().getType())
      return emitOpError("return type mismatch");
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
// EvalOp
//===----------------------------------------------------------------------===//
LogicalResult EvalOp::verify() {
  if (isa<mlir::gccjit::LValueType>(getExpr().getType()))
    return emitOpError("operand should be an rvalue");
  return success();
}
