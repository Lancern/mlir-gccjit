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

#include "mlir-gccjit/Translation/TranslateToGCCJIT.h"
#include "libgccjit.h"
#include "mlir-gccjit/IR/GCCJITAttrs.h"
#include "mlir-gccjit/IR/GCCJITOps.h"
#include "mlir-gccjit/IR/GCCJITOpsEnums.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <utility>
#include <variant>

namespace mlir::gccjit {

namespace {

class Expr {
  std::variant<gcc_jit_lvalue *, gcc_jit_rvalue *> value;

public:
  Expr() : value(static_cast<gcc_jit_lvalue *>(nullptr)) {}
  Expr(gcc_jit_lvalue *value) : value(value) {}
  Expr(gcc_jit_rvalue *value) : value(value) {}
  operator bool() const { return isRValue() || std::get<0>(value) != nullptr; }
  operator gcc_jit_lvalue *() const {
    if (isLValue())
      return std::get<0>(value);
    llvm_unreachable("not an lvalue");
  }
  operator gcc_jit_rvalue *() const {
    if (isRValue())
      return std::get<1>(value);
    return gcc_jit_lvalue_as_rvalue(std::get<0>(value));
  }
  bool isLValue() const { return value.index() == 0; }
  bool isRValue() const { return value.index() == 1; }
};

class RegionVisitor {
  GCCJITTranslation &translator;
  Region &region [[maybe_unused]];
  llvm::DenseMap<Value, Expr> exprCache;
  llvm::DenseMap<Value, gcc_jit_lvalue *> variables;
  llvm::DenseMap<Block *, gcc_jit_block *> blocks;

public:
  RegionVisitor(GCCJITTranslation &translator, Region &region);
  gcc_jit_lvalue *queryVariable(Value value);
  GCCJITTranslation &getTranslator() const;
  gcc_jit_context *getContext() const;
  MLIRContext *getMLIRContext() const;
  void translateIntoContext();

private:
  Expr visitExpr(Operation *op);
  void visitExprAsRValue(ValueRange operands,
                         llvm::SmallVectorImpl<gcc_jit_rvalue *> &result);
  gcc_jit_rvalue *visitExprWithoutCache(ConstantOp op);
  gcc_jit_rvalue *visitExprWithoutCache(LiteralOp op);
  gcc_jit_rvalue *visitExprWithoutCache(SizeOfOp op);
  gcc_jit_rvalue *visitExprWithoutCache(AlignOfOp op);
  gcc_jit_rvalue *visitExprWithoutCache(AsRValueOp op);
  gcc_jit_rvalue *visitExprWithoutCache(BinaryOp op);
  gcc_jit_rvalue *visitExprWithoutCache(UnaryOp op);
  gcc_jit_rvalue *visitExprWithoutCache(CompareOp op);
  gcc_jit_rvalue *visitExprWithoutCache(CallOp op);
  gcc_jit_rvalue *visitExprWithoutCache(CastOp op);
  gcc_jit_rvalue *visitExprWithoutCache(BitCastOp op);
  gcc_jit_rvalue *visitExprWithoutCache(PtrCallOp op);
  gcc_jit_rvalue *visitExprWithoutCache(AddrOp op);
  gcc_jit_rvalue *visitExprWithoutCache(FnAddrOp op);
  gcc_jit_lvalue *visitExprWithoutCache(GetGlobalOp op);
};

} // namespace

void GCCJITTranslation::convertTypes(
    mlir::TypeRange types, llvm::SmallVector<gcc_jit_type *> &result) {
  for (auto type : types)
    result.push_back(convertType(type));
}

gcc_jit_type *GCCJITTranslation::convertType(mlir::Type type) {
  if (auto it = typeMap.find(type); it != typeMap.end())
    return it->second;
  auto *res = llvm::TypeSwitch<mlir::Type, gcc_jit_type *>(type)
                  .Case([&](gccjit::LValueType t) {
                    return convertType(t.getInnerType());
                  })
                  .Case([&](gccjit::PointerType t) {
                    auto *pointee = convertType(t.getElementType());
                    return gcc_jit_type_get_pointer(pointee);
                  })
                  .Case([&](gccjit::QualifiedType t) {
                    auto *res = convertType(t.getElementType());
                    if (t.getIsConst())
                      res = gcc_jit_type_get_const(res);
                    if (t.getIsRestrict())
                      res = gcc_jit_type_get_restrict(res);
                    if (t.getIsVolatile())
                      res = gcc_jit_type_get_volatile(res);
                    return res;
                  })
                  .Case([&](gccjit::IntType t) {
                    auto kind = t.getKind();
                    return gcc_jit_context_get_type(ctxt, kind);
                  })
                  .Case([&](gccjit::FloatType t) {
                    auto kind = t.getKind();
                    return gcc_jit_context_get_type(ctxt, kind);
                  })
                  .Case([&](gccjit::ComplexType t) {
                    auto kind = t.getKind();
                    return gcc_jit_context_get_type(ctxt, kind);
                  })
                  .Case([&](gccjit::VoidType t) {
                    return gcc_jit_context_get_type(ctxt, GCC_JIT_TYPE_VOID);
                  })
                  .Case([&](gccjit::ArrayType t) {
                    auto *elemType = convertType(t.getElementType());
                    auto size = t.getSize();
                    return gcc_jit_context_new_array_type(ctxt, nullptr,
                                                          elemType, size);
                  })
                  .Case([&](gccjit::VectorType t) {
                    auto *elemType = convertType(t.getElementType());
                    auto size = t.getNumUnits();
                    return gcc_jit_type_get_vector(elemType, size);
                  })
                  .Default([](mlir::Type) { return nullptr; });
  typeMap[type] = res;
  return res;
}

GCCJITTranslation::GCCJITTranslation() : ctxt(gcc_jit_context_acquire()) {}

GCCJITTranslation::~GCCJITTranslation() {
  if (ctxt) {
    gcc_jit_context_release(ctxt);
  }
}

GCCJITContext GCCJITTranslation::takeContext() {
  return GCCJITContext(std::exchange(ctxt, nullptr));
}

gcc_jit_context *GCCJITTranslation::getContext() const { return ctxt; }

void GCCJITTranslation::translateModuleToGCCJIT(ModuleOp op) {
  moduleOp = op;
  populateGCCJITModuleOptions();
  declareAllFunctionAndGlobals();
  translateGlobalInitializers();
}

gcc_jit_location *GCCJITTranslation::getLocation(LocationAttr loc) {
  if (!loc)
    return nullptr;
  return llvm::TypeSwitch<LocationAttr, gcc_jit_location *>(loc)
      .Case([&](FileLineColLoc loc) {
        return gcc_jit_context_new_location(ctxt,
                                            loc.getFilename().str().c_str(),
                                            loc.getLine(), loc.getColumn());
      })
      .Case([&](CallSiteLoc loc) { return getLocation(loc.getCaller()); })
      .Case(
          [&](FusedLoc loc) { return getLocation(loc.getLocations().front()); })
      .Case([&](NameLoc loc) { return getLocation(loc.getChildLoc()); })
      .Case(
          [&](OpaqueLoc loc) { return getLocation(loc.getFallbackLocation()); })
      .Case([&](UnknownLoc loc) { return nullptr; })
      .Default([](LocationAttr) { return nullptr; });
}

gcc_jit_lvalue *GCCJITTranslation::getGlobalLValue(SymbolRefAttr symbol) {
  return globalMap.lookup(symbol);
}

gcc_jit_function *GCCJITTranslation::getFunction(SymbolRefAttr symbol) {
  return functionMap.lookup(symbol).fnHandle;
}

void GCCJITTranslation::populateGCCJITModuleOptions() {
  for (auto &attr : moduleOp->getAttrs()) {
    if (attr.getName() == "gccjit.prog_name") {
      if (auto strAttr = dyn_cast<StringAttr>(attr.getValue()))
        gcc_jit_context_set_str_option(ctxt, GCC_JIT_STR_OPTION_PROGNAME,
                                       strAttr.str().c_str());
    } else if (attr.getName() == "gccjit.opt_level") {
      if (auto intAttr = dyn_cast<OptLevelAttr>(attr.getValue())) {
        int optLevel = static_cast<int>(intAttr.getLevel().getValue());
        gcc_jit_context_set_int_option(
            ctxt, GCC_JIT_INT_OPTION_OPTIMIZATION_LEVEL, optLevel);
      }
    } else if (attr.getName() == "gccjit.allow_unreachable") {
      if (auto boolAttr = dyn_cast<BoolAttr>(attr.getValue()))
        gcc_jit_context_set_bool_allow_unreachable_blocks(ctxt,
                                                          boolAttr.getValue());
    } else if (attr.getName() == "gccjit.debug_info") {
      if (auto boolAttr = dyn_cast<BoolAttr>(attr.getValue()))
        gcc_jit_context_set_bool_option(ctxt, GCC_JIT_BOOL_OPTION_DEBUGINFO,
                                        boolAttr.getValue());
    }
  }
}

static gcc_jit_function_kind convertFnKind(FnKind kind) {
  switch (kind) {
  case FnKind::Exported:
    return GCC_JIT_FUNCTION_EXPORTED;
  case FnKind::Internal:
    return GCC_JIT_FUNCTION_INTERNAL;
  case FnKind::Imported:
    return GCC_JIT_FUNCTION_IMPORTED;
  case FnKind::AlwaysInline:
    return GCC_JIT_FUNCTION_ALWAYS_INLINE;
  }
  llvm_unreachable("unknown function kind");
}

static void processFunctionAttrs(gccjit::FuncOp func,
                                 gcc_jit_function *handle) {
  for (auto attr : func.getGccjitFnAttrs()) {
    auto fnAttr = cast<FunctionAttr>(attr);
    switch (fnAttr.getAttr().getValue()) {
    case FnAttrEnum::Alias:
      gcc_jit_function_add_string_attribute(handle, GCC_JIT_FN_ATTRIBUTE_ALIAS,
                                            fnAttr.getStrValue().str().c_str());
      break;
    case FnAttrEnum::AlwaysInline:
      gcc_jit_function_add_attribute(handle,
                                     GCC_JIT_FN_ATTRIBUTE_ALWAYS_INLINE);
      break;
    case FnAttrEnum::Inline:
      gcc_jit_function_add_attribute(handle, GCC_JIT_FN_ATTRIBUTE_INLINE);
      break;
    case FnAttrEnum::NoInline:
      gcc_jit_function_add_attribute(handle, GCC_JIT_FN_ATTRIBUTE_NOINLINE);
      break;
    case FnAttrEnum::Target:
      gcc_jit_function_add_string_attribute(handle, GCC_JIT_FN_ATTRIBUTE_TARGET,
                                            fnAttr.getStrValue().str().c_str());
      break;
    case FnAttrEnum::Used:
      gcc_jit_function_add_attribute(handle, GCC_JIT_FN_ATTRIBUTE_USED);
      break;
    case FnAttrEnum::Visibility:
      gcc_jit_function_add_string_attribute(handle,
                                            GCC_JIT_FN_ATTRIBUTE_VISIBILITY,
                                            fnAttr.getStrValue().str().c_str());
      break;
    case FnAttrEnum::Cold:
      gcc_jit_function_add_attribute(handle, GCC_JIT_FN_ATTRIBUTE_COLD);
      break;
    case FnAttrEnum::ReturnsTwice:
      gcc_jit_function_add_attribute(handle,
                                     GCC_JIT_FN_ATTRIBUTE_RETURNS_TWICE);
      break;
    case FnAttrEnum::Pure:
      gcc_jit_function_add_attribute(handle, GCC_JIT_FN_ATTRIBUTE_PURE);
      break;
    case FnAttrEnum::Const:
      gcc_jit_function_add_attribute(handle, GCC_JIT_FN_ATTRIBUTE_CONST);
      break;
    case FnAttrEnum::Weak:
      gcc_jit_function_add_attribute(handle, GCC_JIT_FN_ATTRIBUTE_WEAK);
      break;
    case FnAttrEnum::Nonnull:
      gcc_jit_function_add_integer_array_attribute(
          handle, GCC_JIT_FN_ATTRIBUTE_NONNULL,
          reinterpret_cast<const int *>(
              fnAttr.getIntArrayValue().asArrayRef().data()),
          fnAttr.getIntArrayValue().size());
      break;
    }
  }
}

static gcc_jit_global_kind convertGlobalKind(GlbKind kind) {
  switch (kind) {
  case GlbKind::Exported:
    return GCC_JIT_GLOBAL_EXPORTED;
  case GlbKind::Internal:
    return GCC_JIT_GLOBAL_INTERNAL;
  case GlbKind::Imported:
    return GCC_JIT_GLOBAL_IMPORTED;
  }
  llvm_unreachable("unknown global kind");
}

static gcc_jit_tls_model convertTLSModel(TLSModelEnum model) {
  switch (model) {
  case TLSModelEnum::GlobalDynamic:
    return GCC_JIT_TLS_MODEL_GLOBAL_DYNAMIC;
  case TLSModelEnum::LocalDynamic:
    return GCC_JIT_TLS_MODEL_LOCAL_DYNAMIC;
  case TLSModelEnum::InitialExec:
    return GCC_JIT_TLS_MODEL_INITIAL_EXEC;
  case TLSModelEnum::LocalExec:
    return GCC_JIT_TLS_MODEL_LOCAL_EXEC;
  case TLSModelEnum::None:
    return GCC_JIT_TLS_MODEL_NONE;
  }
  llvm_unreachable("unknown TLS model");
}

void GCCJITTranslation::declareAllFunctionAndGlobals() {
  moduleOp.walk([&](gccjit::FuncOp func) {
    auto type = func.getFunctionType();
    llvm::SmallVector<gcc_jit_type *> paramTypes;
    llvm::SmallVector<gcc_jit_param *> params;
    convertTypes(type.getInputs(), paramTypes);
    auto *returnType = convertType(type.getReturnType());
    auto kind = convertFnKind(func.getFnKind());
    auto name = func.getSymName().str();
    auto enumerated = llvm::enumerate(paramTypes);
    std::transform(enumerated.begin(), enumerated.end(),
                   std::back_inserter(params), [&](auto pair) {
                     auto index = pair.index();
                     auto type = pair.value();
                     auto name =
                         llvm::Twine("arg").concat(llvm::Twine(index)).str();
                     return gcc_jit_context_new_param(
                         ctxt, /*todo: location*/ nullptr, type, name.c_str());
                   });
    auto *funcHandle = gcc_jit_context_new_function(
        ctxt, getLocation(func.getLoc()), kind, returnType, name.c_str(),
        paramTypes.size(), params.data(), type.isVarArg());
    processFunctionAttrs(func, funcHandle);
    SymbolRefAttr symRef = SymbolRefAttr::get(getMLIRContext(), name);
    functionMap[symRef] = {funcHandle, std::move(params)};
  });
  moduleOp.walk([&](gccjit::GlobalOp global) {
    auto type = global.getType();
    auto *typeHandle = convertType(type);
    auto name = global.getSymName().str();
    auto nameAttr = SymbolRefAttr::get(getMLIRContext(), name);
    auto kind = convertGlobalKind(global.getGlbKind());
    auto *globalHandle = gcc_jit_context_new_global(
        ctxt, getLocation(global.getLoc()), kind, typeHandle, name.c_str());
    globalMap[nameAttr] = globalHandle;
    if (auto regName = global.getRegName())
      gcc_jit_lvalue_set_register_name(globalHandle,
                                       regName->getName().str().c_str());
    if (auto alignment = global.getAlignment())
      gcc_jit_lvalue_set_alignment(globalHandle, alignment->getZExtValue());
    if (auto tlsModel = global.getTlsModel())
      gcc_jit_lvalue_set_tls_model(
          globalHandle, convertTLSModel(tlsModel->getModel().getValue()));
    if (auto linkSection = global.getLinkSection())
      gcc_jit_lvalue_set_link_section(globalHandle,
                                      linkSection->getSection().str().c_str());
    if (auto visibility = global.getVisibility())
      gcc_jit_lvalue_add_string_attribute(
          globalHandle, GCC_JIT_VARIABLE_ATTRIBUTE_VISIBILITY,
          visibility->getVisibility().str().c_str());
    if (auto initializer = global.getInitializer()) {
      llvm::TypeSwitch<Attribute>(*initializer)
          .Case([&](StringLiteralAttr attr) {
            auto str = attr.getInitializer();
            auto blob = str.str();
            gcc_jit_global_set_initializer(globalHandle, blob.c_str(),
                                           blob.size() + 1);
          })
          .Case([&](ByteArrayInitializerAttr attr) {
            auto data = attr.getInitializer().asArrayRef();
            gcc_jit_global_set_initializer(globalHandle, data.data(),
                                           data.size());
          })
          .Default([](Attribute) { llvm_unreachable("unknown initializer"); });
    }
    // if the global has body, we translate them in the next pass
  });
}

void GCCJITTranslation::translateGlobalInitializers() {
  moduleOp.walk([&](gccjit::GlobalOp global) {
    if (global.getBody().empty())
      return;
    RegionVisitor visitor(*this, global.getBody());
    visitor.translateIntoContext();
  });
}

///===----------------------------------------------------------------------===///
/// RegionVisitor
///===----------------------------------------------------------------------===///

RegionVisitor::RegionVisitor(GCCJITTranslation &translator, Region &region)
    : translator(translator), region(region) {
  size_t variableCount = 0;
  if (auto funcOp = dyn_cast<gccjit::FuncOp>(region.getParentOp())) {
    auto symName = SymbolRefAttr::get(funcOp.getOperation()->getContext(),
                                      funcOp.getSymName());
    auto *function = translator.getFunction(symName);
    for (auto arg : region.getArguments()) {
      auto *lvalue = gcc_jit_function_get_param(function, arg.getArgNumber());
      variables[arg] = gcc_jit_param_as_lvalue(lvalue);
    }
    region.walk([&](LocalOp local) {
      auto *type = translator.convertType(local.getType());
      auto *loc = translator.getLocation(local.getLoc());
      auto name = llvm::Twine("var").concat(llvm::Twine(variableCount++)).str();
      auto *lvalue =
          gcc_jit_function_new_local(function, loc, type, name.c_str());
      variables[local] = lvalue;
    });
  }
}

gcc_jit_lvalue *RegionVisitor::queryVariable(Value value) {
  return variables.lookup(value);
}

GCCJITTranslation &RegionVisitor::getTranslator() const { return translator; }

gcc_jit_context *RegionVisitor::getContext() const {
  return translator.getContext();
}

MLIRContext *RegionVisitor::getMLIRContext() const {
  return translator.getMLIRContext();
}

void RegionVisitor::translateIntoContext() {
  auto *parent = region.getParentOp();
  if (auto funcOp = dyn_cast<gccjit::FuncOp>(parent)) {
    (void)funcOp;
    llvm_unreachable("NYI");
  }
  if (auto globalOp = dyn_cast<gccjit::GlobalOp>(parent)) {
    assert(region.getBlocks().size() == 1 &&
           "global initializer region should have a single block");
    Block &block = region.getBlocks().front();
    auto terminator = cast<gccjit::ReturnOp>(block.getTerminator());
    auto value = terminator->getOperand(0);
    auto rvalue = visitExpr(value.getDefiningOp());
    auto symName = SymbolRefAttr::get(getMLIRContext(), globalOp.getSymName());
    auto *lvalue = getTranslator().getGlobalLValue(symName);
    gcc_jit_global_set_initializer_rvalue(lvalue, rvalue);
    return;
  }
  llvm_unreachable("unknown region parent");
}

Expr RegionVisitor::visitExpr(Operation *op) {
  if (op->getNumResults() != 1)
    llvm_unreachable("expected single result operation");

  auto &cached = exprCache[op->getResult(0)];
  if (!cached)
    cached =
        llvm::TypeSwitch<Operation *, Expr>(op)
            .Case([&](ConstantOp op) { return visitExprWithoutCache(op); })
            .Case([&](LiteralOp op) { return visitExprWithoutCache(op); })
            .Case([&](SizeOfOp op) { return visitExprWithoutCache(op); })
            .Case([&](AlignOfOp op) { return visitExprWithoutCache(op); })
            .Case([&](AsRValueOp op) { return visitExprWithoutCache(op); })
            .Case([&](BinaryOp op) { return visitExprWithoutCache(op); })
            .Case([&](UnaryOp op) { return visitExprWithoutCache(op); })
            .Case([&](CompareOp op) { return visitExprWithoutCache(op); })
            .Case([&](CallOp op) { return visitExprWithoutCache(op); })
            .Case([&](CastOp op) { return visitExprWithoutCache(op); })
            .Case([&](BitCastOp op) { return visitExprWithoutCache(op); })
            .Case([&](PtrCallOp op) { return visitExprWithoutCache(op); })
            .Case([&](AddrOp op) { return visitExprWithoutCache(op); })
            .Case([&](FnAddrOp op) { return visitExprWithoutCache(op); })
            .Case([&](GetGlobalOp op) { return visitExprWithoutCache(op); })
            .Default([](Operation *) -> Expr {
              llvm_unreachable("unknown expression type");
            });
  return cached;
}

void RegionVisitor::visitExprAsRValue(
    ValueRange operands, llvm::SmallVectorImpl<gcc_jit_rvalue *> &result) {
  for (auto operand : operands)
    result.push_back(visitExpr(operand.getDefiningOp()));
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(ConstantOp op) {
  auto type = op.getType();
  auto *typeHandle = getTranslator().convertType(type);
  return llvm::TypeSwitch<TypedAttr, gcc_jit_rvalue *>(op.getValue())
      .Case([&](ZeroAttr attr) {
        return gcc_jit_context_zero(getContext(), typeHandle);
      })
      .Case([&](NullAttr attr) {
        return gcc_jit_context_null(getContext(), typeHandle);
      })
      .Case([&](OneAttr attr) {
        return gcc_jit_context_one(getContext(), typeHandle);
      })
      .Case([&](IntAttr attr) {
        // TODO: handle signedness and width
        auto value = attr.getValue();
        return gcc_jit_context_new_rvalue_from_long(getContext(), typeHandle,
                                                    value.getZExtValue());
      })
      .Case([&](FloatAttr attr) {
        auto value = attr.getValue();
        return gcc_jit_context_new_rvalue_from_double(getContext(), typeHandle,
                                                      value.convertToDouble());
      })
      .Default([](TypedAttr) -> gcc_jit_rvalue * {
        llvm_unreachable("unknown constant type");
      });
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(LiteralOp op) {
  auto string = op.getValue().getInitializer().str();
  return gcc_jit_context_new_string_literal(getContext(), string.c_str());
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(SizeOfOp op) {
  auto type = op.getType();
  auto *typeHandle = getTranslator().convertType(type);
  return gcc_jit_context_new_sizeof(getContext(), typeHandle);
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(AlignOfOp op) {
  llvm_unreachable("GCCJIT does not support alignof yet");
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(AsRValueOp op) {
  auto lvalue = visitExpr(op.getLvalue().getDefiningOp());
  return gcc_jit_lvalue_as_rvalue(lvalue);
}

static gcc_jit_binary_op convertBinaryOp(BOp kind) {
  switch (kind) {
  case BOp::Plus:
    return GCC_JIT_BINARY_OP_PLUS;
  case BOp::Minus:
    return GCC_JIT_BINARY_OP_MINUS;
  case BOp::Mult:
    return GCC_JIT_BINARY_OP_MULT;
  case BOp::Divide:
    return GCC_JIT_BINARY_OP_DIVIDE;
  case BOp::Modulo:
    return GCC_JIT_BINARY_OP_MODULO;
  case BOp::BitwiseAnd:
    return GCC_JIT_BINARY_OP_BITWISE_AND;
  case BOp::BitwiseXor:
    return GCC_JIT_BINARY_OP_BITWISE_XOR;
  case BOp::BitwiseOr:
    return GCC_JIT_BINARY_OP_BITWISE_OR;
  case BOp::LogicalAnd:
    return GCC_JIT_BINARY_OP_LOGICAL_AND;
  case BOp::LogicalOr:
    return GCC_JIT_BINARY_OP_LOGICAL_OR;
  case BOp::LShift:
    return GCC_JIT_BINARY_OP_LSHIFT;
  case BOp::RShift:
    return GCC_JIT_BINARY_OP_RSHIFT;
  }
  llvm_unreachable("unknown binary operation");
}

// RValue always has a defining operation
gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(BinaryOp op) {
  auto lhs = visitExpr(op.getLhs().getDefiningOp());
  auto rhs = visitExpr(op.getRhs().getDefiningOp());
  auto kind = convertBinaryOp(op.getOp());
  auto *loc = getTranslator().getLocation(op.getLoc());
  auto *ctxt = getContext();
  auto *type = getTranslator().convertType(op.getType());
  return gcc_jit_context_new_binary_op(ctxt, loc, kind, type, lhs, rhs);
}

static gcc_jit_unary_op convertUnaryOp(UOp kind) {
  switch (kind) {
  case UOp::Minus:
    return GCC_JIT_UNARY_OP_MINUS;
  case UOp::BitwiseNegate:
    return GCC_JIT_UNARY_OP_BITWISE_NEGATE;
  case UOp::LogicalNegate:
    return GCC_JIT_UNARY_OP_LOGICAL_NEGATE;
  case UOp::Abs:
    return GCC_JIT_UNARY_OP_ABS;
  }
  llvm_unreachable("unknown unary operation");
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(UnaryOp op) {
  auto operand = visitExpr(op.getOperand().getDefiningOp());
  auto kind = convertUnaryOp(op.getOp());
  auto *loc = getTranslator().getLocation(op.getLoc());
  auto *ctxt = getContext();
  auto *type = getTranslator().convertType(op.getType());
  return gcc_jit_context_new_unary_op(ctxt, loc, kind, type, operand);
}

static gcc_jit_comparison convertCompareOp(CmpOp kind) {
  switch (kind) {
  case CmpOp::Eq:
    return GCC_JIT_COMPARISON_EQ;
  case CmpOp::Ne:
    return GCC_JIT_COMPARISON_NE;
  case CmpOp::Lt:
    return GCC_JIT_COMPARISON_LT;
  case CmpOp::Le:
    return GCC_JIT_COMPARISON_LE;
  case CmpOp::Gt:
    return GCC_JIT_COMPARISON_GT;
  case CmpOp::Ge:
    return GCC_JIT_COMPARISON_GE;
  }
  llvm_unreachable("unknown compare operation");
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(CompareOp op) {
  auto lhs = visitExpr(op.getLhs().getDefiningOp());
  auto rhs = visitExpr(op.getRhs().getDefiningOp());
  auto kind = convertCompareOp(op.getOp());
  auto *loc = getTranslator().getLocation(op.getLoc());
  auto *ctxt = getContext();
  return gcc_jit_context_new_comparison(ctxt, loc, kind, lhs, rhs);
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(CallOp op) {
  gcc_jit_function *callee = nullptr;
  if (op.getBuiltin()) {
    callee = gcc_jit_context_get_builtin_function(
        getContext(), op.getCallee().getLeafReference().str().c_str());
  } else {
    callee = getTranslator().getFunction(op.getCallee());
  }
  assert(callee && "function not found");
  llvm::SmallVector<gcc_jit_rvalue *> args;
  visitExprAsRValue(op.getArgs(), args);
  auto *loc = getTranslator().getLocation(op.getLoc());
  auto *ctxt = getContext();
  auto *call =
      gcc_jit_context_new_call(ctxt, loc, callee, args.size(), args.data());
  gcc_jit_rvalue_set_bool_require_tail_call(call, op.getTail());
  return call;
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(CastOp op) {
  auto operand = visitExpr(op.getOperand().getDefiningOp());
  auto *loc = getTranslator().getLocation(op.getLoc());
  auto *ctxt = getContext();
  auto *type = getTranslator().convertType(op.getType());
  return gcc_jit_context_new_cast(ctxt, loc, operand, type);
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(BitCastOp op) {
  auto operand = visitExpr(op.getOperand().getDefiningOp());
  auto *loc = getTranslator().getLocation(op.getLoc());
  auto *ctxt = getContext();
  auto *type = getTranslator().convertType(op.getType());
  return gcc_jit_context_new_bitcast(ctxt, loc, operand, type);
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(PtrCallOp op) {
  auto callee = visitExpr(op.getCallee().getDefiningOp());
  llvm::SmallVector<gcc_jit_rvalue *> args;
  visitExprAsRValue(op.getArgs(), args);
  auto *loc = getTranslator().getLocation(op.getLoc());
  auto *ctxt = getContext();
  auto *call = gcc_jit_context_new_call_through_ptr(ctxt, loc, callee,
                                                    args.size(), args.data());
  gcc_jit_rvalue_set_bool_require_tail_call(call, op.getTail());
  return call;
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(AddrOp op) {
  auto lvalue = visitExpr(op.getOperand().getDefiningOp());
  auto *loc = getTranslator().getLocation(op.getLoc());
  return gcc_jit_lvalue_get_address(lvalue, loc);
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(FnAddrOp op) {
  auto *fn = getTranslator().getFunction(op.getCallee());
  assert(fn && "function not found");
  auto *loc = getTranslator().getLocation(op.getLoc());
  return gcc_jit_function_get_address(fn, loc);
}

gcc_jit_lvalue *RegionVisitor::visitExprWithoutCache(GetGlobalOp op) {
  auto *lvalue = getTranslator().getGlobalLValue(op.getSym());
  assert(lvalue && "global not found");
  return lvalue;
}

//===----------------------------------------------------------------------===//
// TranslateModuleToGCCJIT
//===----------------------------------------------------------------------===//
llvm::Expected<GCCJITContext> translateModuleToGCCJIT(ModuleOp op) {
  GCCJITTranslation translator;
  translator.translateModuleToGCCJIT(op);
  return translator.takeContext();
}

///===----------------------------------------------------------------------===///
/// GCCJITContextDeleter
///===----------------------------------------------------------------------===///
void GCCJITContextDeleter::operator()(gcc_jit_context *ctxt) const {
  if (ctxt)
    gcc_jit_context_release(ctxt);
}

} // namespace mlir::gccjit
