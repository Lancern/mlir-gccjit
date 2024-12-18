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

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <utility>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>

#include <libgccjit.h>

#include "mlir-gccjit/IR/GCCJITAttrs.h"
#include "mlir-gccjit/IR/GCCJITOps.h"
#include "mlir-gccjit/IR/GCCJITOpsEnums.h"
#include "mlir-gccjit/IR/GCCJITTypes.h"

namespace mlir::gccjit {

namespace {

class Expr {
  static constexpr intptr_t rvalueFlag = std::numeric_limits<intptr_t>::min();
  union {
    gcc_jit_lvalue *lvalue;
    gcc_jit_rvalue *rvalue;
    intptr_t dummy;
  };

public:
  Expr() : dummy(0) {}
  Expr(gcc_jit_lvalue *value) : lvalue(value) {}
  Expr(gcc_jit_rvalue *value) : rvalue(value) { dummy |= rvalueFlag; }
  operator bool() const { return dummy != 0; }
  operator gcc_jit_lvalue *() const {
    if (isLValue())
      return lvalue;
    llvm_unreachable("not an lvalue");
  }
  operator gcc_jit_rvalue *() const {
    if (isRValue())
      return reinterpret_cast<gcc_jit_rvalue *>(dummy & ~rvalueFlag);
    return gcc_jit_lvalue_as_rvalue(lvalue);
  }
  bool isLValue() const { return dummy > 0; }
  bool isRValue() const { return dummy < 0; }
};

class RegionVisitor {
  GCCJITTranslation &translator;
  Region &region [[maybe_unused]];
  llvm::DenseMap<Value, Expr> exprCache;
  llvm::DenseMap<Block *, gcc_jit_block *> blocks;
  bool valueMaterialization;
  RegionVisitor *parent;

public:
  RegionVisitor(GCCJITTranslation &translator, Region &region,
                RegionVisitor *parent = nullptr);
  GCCJITTranslation &getTranslator() const;
  gcc_jit_context *getContext() const;
  MLIRContext *getMLIRContext() const;
  Expr translateIntoContext();

private:
  Expr visitExpr(Value value, bool toplevel = false);
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
  gcc_jit_rvalue *visitExprWithoutCache(NewStructOp op);
  gcc_jit_rvalue *visitExprWithoutCache(NewArrayOp op);
  gcc_jit_rvalue *visitExprWithoutCache(NewUnionOp op);
  gcc_jit_rvalue *visitExprWithoutCache(NewVectorOp op);
  gcc_jit_lvalue *visitExprWithoutCache(GetGlobalOp op);
  Expr visitExprWithoutCache(ExprOp op);
  gcc_jit_lvalue *visitExprWithoutCache(DerefOp op);
  Expr visitExprWithoutCache(AccessFieldOp op);
  gcc_jit_lvalue *visitExprWithoutCache(DerefFieldOp op);
  gcc_jit_rvalue *visitExprWithoutCache(AtomicLoadOp op);
  gcc_jit_rvalue *visitExprWithoutCache(AtomicRMWOp op);
  gcc_jit_rvalue *visitExprWithoutCache(AtomicCompareExchangeOp op);

  /// The following operations are entrypoints for real codegen.
  void visitAssignOp(gcc_jit_block *blk, AssignOp op);
  void visitUpdateOp(gcc_jit_block *blk, UpdateOp op);
  void visitReturnOp(gcc_jit_block *blk, ReturnOp op);
  void visitSwitchOp(gcc_jit_block *blk, SwitchOp op);
  void visitJumpOp(gcc_jit_block *blk, JumpOp op);
  void visitAsmOp(gcc_jit_block *blk, AsmOp op);
  void visitAsmGotoOp(gcc_jit_block *blk, AsmGotoOp op);
  void visitConditionalOp(gcc_jit_block *blk, ConditionalOp op);
  void visitAtomicStoreOp(gcc_jit_block *blk, AtomicStoreOp op);

  gcc_jit_block *lookupBlock(Block *block);
  Expr &lookupExpr(Value value);
  Expr *lookupExprImpl(Value value);

  template <typename OpTy>
  void populateExtendedAsm(gcc_jit_extended_asm *extAsm, OpTy op);
};

} // namespace

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
  translateFunctions();
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

gcc_jit_location *GCCJITTranslation::convertLocation(SourceLocAttr loc) {
  if (!loc)
    return nullptr;
  return gcc_jit_context_new_location(ctxt, loc.getFilename().str().c_str(),
                                      loc.getLine(), loc.getColumn());
}

gcc_jit_lvalue *GCCJITTranslation::getGlobalLValue(SymbolRefAttr symbol) {
  return globalMap.lookup(symbol);
}

GCCJITTranslation::FunctionEntry
GCCJITTranslation::getFunction(SymbolRefAttr symbol) {
  return functionMap.lookup(symbol);
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
    } else if (attr.getName() == "gccjit.cmdline_options") {
      if (auto arrayAttr = dyn_cast<ArrayAttr>(attr.getValue()))
        for (auto &element : arrayAttr.getValue())
          if (auto strAttr = dyn_cast<StringAttr>(element))
            gcc_jit_context_add_command_line_option(
                ctxt, strAttr.getValue().str().c_str());
    } else if (attr.getName() == "gccjit.driver_options") {
      if (auto arrayAttr = dyn_cast<ArrayAttr>(attr.getValue()))
        for (auto &element : arrayAttr.getValue())
          if (auto strAttr = dyn_cast<StringAttr>(element))
            gcc_jit_context_add_driver_option(ctxt,
                                              strAttr.getValue().str().c_str());
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
      gcc_jit_function_add_string_attribute(
          handle, GCC_JIT_FN_ATTRIBUTE_ALIAS,
          fnAttr.getStrValue().value().str().c_str());
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
      gcc_jit_function_add_string_attribute(
          handle, GCC_JIT_FN_ATTRIBUTE_TARGET,
          fnAttr.getStrValue().value().str().c_str());
      break;
    case FnAttrEnum::Used:
      gcc_jit_function_add_attribute(handle, GCC_JIT_FN_ATTRIBUTE_USED);
      break;
    case FnAttrEnum::Visibility:
      gcc_jit_function_add_string_attribute(
          handle, GCC_JIT_FN_ATTRIBUTE_VISIBILITY,
          fnAttr.getStrValue().value().str().c_str());
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
              fnAttr.getIntArrayValue().value().asArrayRef().data()),
          fnAttr.getIntArrayValue().value().size());
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
  for (auto func : moduleOp.getOps<gccjit::FuncOp>()) {
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
                         llvm::Twine("%arg").concat(llvm::Twine(index)).str();
                     return gcc_jit_context_new_param(
                         ctxt, /*todo: location*/ nullptr, type, name.c_str());
                   });
    auto *funcHandle = gcc_jit_context_new_function(
        ctxt, getLocation(func.getLoc()), kind, returnType, name.c_str(),
        paramTypes.size(), params.data(), type.isVarArg());
    processFunctionAttrs(func, funcHandle);
    SymbolRefAttr symRef = SymbolRefAttr::get(getMLIRContext(), name);
    functionMap.try_emplace(symRef, funcHandle);
  }
  for (auto global : moduleOp.getOps<gccjit::GlobalOp>()) {
    auto type = global.getType();
    auto *typeHandle = convertType(type);
    auto name = global.getSymName().str();
    auto nameAttr = SymbolRefAttr::get(getMLIRContext(), name);
    auto kind = convertGlobalKind(global.getGlbKind());
    auto *globalHandle = gcc_jit_context_new_global(
        ctxt, getLocation(global.getLoc()), kind, typeHandle, name.c_str());
    globalMap[nameAttr] = globalHandle;
    if (auto regName = global.getRegName())
      gcc_jit_lvalue_set_register_name(globalHandle, regName->str().c_str());
    if (auto alignment = global.getAlignment())
      gcc_jit_lvalue_set_alignment(globalHandle, alignment->getZExtValue());
    if (auto tlsModel = global.getTlsModel())
      gcc_jit_lvalue_set_tls_model(
          globalHandle, convertTLSModel(tlsModel->getModel().getValue()));
    if (auto linkSection = global.getLinkSection())
      gcc_jit_lvalue_set_link_section(globalHandle, linkSection->str().c_str());
    if (auto visibility = global.getVisibility())
      gcc_jit_lvalue_add_string_attribute(globalHandle,
                                          GCC_JIT_VARIABLE_ATTRIBUTE_VISIBILITY,
                                          visibility->str().c_str());

    if ([[maybe_unused]] auto readonly = global.getReadonly()) {
#ifdef LIBGCCJIT_HAVE_gcc_jit_global_set_readonly
      gcc_jit_global_set_readonly(globalHandle);
#else
      llvm_unreachable("gcc_jit_global_set_readonly is not available");
#endif
    }
    if (auto initializer = global.getInitializer()) {
      llvm::TypeSwitch<Attribute>(*initializer)
          .Case([&](StringAttr attr) {
            auto blob = attr.str();
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
  }
}

void GCCJITTranslation::translateGlobalInitializers() {
  for (auto global : moduleOp.getOps<gccjit::GlobalOp>()) {
    if (global.getBody().empty())
      return;
    RegionVisitor visitor(*this, global.getBody());
    visitor.translateIntoContext();
  }
}

///===----------------------------------------------------------------------===///
/// RegionVisitor
///===----------------------------------------------------------------------===///

RegionVisitor::RegionVisitor(GCCJITTranslation &translator, Region &region,
                             RegionVisitor *parent)
    : translator(translator), region(region), parent(parent) {
  auto funcOp = llvm::dyn_cast_if_present<gccjit::FuncOp>(region.getParentOp());
  if (!funcOp) {
    valueMaterialization = false;
    return;
  }

  valueMaterialization = true;

  auto symName = SymbolRefAttr::get(funcOp.getOperation()->getContext(),
                                    funcOp.getSymName());
  auto function = translator.getFunction(symName);
  for (auto arg : region.getArguments()) {
    auto *lvalue = gcc_jit_function_get_param(function, arg.getArgNumber());
    exprCache[arg] = gcc_jit_param_as_lvalue(lvalue);
  }
  AsmState asmState(funcOp);
  region.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op->getNumResults() != 1)
      return WalkResult::skip();

    auto res = op->getResult(0);

    if (auto globalOp = dyn_cast<gccjit::GetGlobalOp>(op)) {
      auto *lvalue = translator.getGlobalLValue(globalOp.getSymAttr());
      exprCache[res] = lvalue;
      return WalkResult::skip();
    }

    // lvalue/void expressions are never materialized
    if (isa<LValueType, VoidType>(op->getResult(0).getType()) &&
        !isa<LocalOp>(op))
      return WalkResult::skip();

    // skip lazy evaluated expressions
    if (auto exprOp = dyn_cast<ExprOp>(op))
      if (exprOp.getLazy())
        return WalkResult::skip();

    auto *type = translator.convertType(res.getType());
    auto *loc = translator.getLocation(res.getLoc());
    std::string name;
    auto varName = op->getAttrOfType<StringAttr>("gccjit.var_name");
    if (!varName) {
      llvm::raw_string_ostream bufferStream(name);
      res.printAsOperand(bufferStream, asmState);
      bufferStream.flush();
    } else {
      name = varName.getValue().str();
    }
    auto *lvalue =
        gcc_jit_function_new_local(function, loc, type, name.c_str());
    exprCache[res] = lvalue;
    return isa<ExprOp>(op) ? WalkResult::skip() : WalkResult::advance();
  });

  unsigned nextBlockId = 0;
  auto nextBlockName = [&] {
    std::string name;
    llvm::raw_string_ostream nameWriter(name);
    nameWriter << "bb" << nextBlockId++;
    nameWriter.flush();
    return name;
  };

  for (auto &block : region) {
    std::string name = nextBlockName();
    auto *blk = gcc_jit_function_new_block(function, name.c_str());
    blocks[&block] = blk;
  }
}

GCCJITTranslation &RegionVisitor::getTranslator() const { return translator; }

gcc_jit_context *RegionVisitor::getContext() const {
  return translator.getContext();
}

MLIRContext *RegionVisitor::getMLIRContext() const {
  return translator.getMLIRContext();
}

Expr RegionVisitor::translateIntoContext() {
  auto *parent = region.getParentOp();
  if (auto funcOp = dyn_cast<gccjit::FuncOp>(parent)) {
    for (auto [mlirBlk, gccBlk] : blocks) {
      auto *blk = gccBlk;
      for (auto &op : *mlirBlk) {
        llvm::TypeSwitch<Operation *>(&op)
            .Case([&](AssignOp op) { visitAssignOp(blk, op); })
            .Case([&](UpdateOp op) { visitUpdateOp(blk, op); })
            .Case([&](ReturnOp op) { visitReturnOp(blk, op); })
            .Case([&](SwitchOp op) { visitSwitchOp(blk, op); })
            .Case([&](JumpOp op) { visitJumpOp(blk, op); })
            .Case([&](AsmOp op) { visitAsmOp(blk, op); })
            .Case([&](AsmGotoOp op) { visitAsmGotoOp(blk, op); })
            .Case([&](ConditionalOp op) { visitConditionalOp(blk, op); })
            .Case([&](AtomicStoreOp op) { visitAtomicStoreOp(blk, op); })
            .Default([&](Operation *op) {
              if (valueMaterialization) {
                auto *loc = translator.getLocation(op->getLoc());
                if (op->getNumResults() == 1 &&
                    !isa<LValueType, VoidType>(op->getResult(0).getType())) {
                  if (auto exprOp = dyn_cast<ExprOp>(op))
                    if (exprOp.getLazy())
                      return;
                  auto result = op->getResult(0);
                  auto rvalue = visitExpr(result, true);
                  auto lvalue = lookupExpr(result);
                  gcc_jit_block_add_assignment(blk, loc, lvalue, rvalue);
                } else if (auto callOp = dyn_cast<CallOp>(op)) {
                  auto *funcCall = visitExprWithoutCache(callOp);
                  gcc_jit_block_add_eval(blk, loc, funcCall);
                } else if (auto ptrCallOp = dyn_cast<PtrCallOp>(op)) {
                  auto *funcCall = visitExprWithoutCache(ptrCallOp);
                  gcc_jit_block_add_eval(blk, loc, funcCall);
                }
              }
            });
      }
    }
    return {};
  }

  assert(mlir::isa<ExprOp>(parent) || mlir::isa<GlobalOp>(parent));
  assert(region.getBlocks().size() == 1 && !valueMaterialization);
  Block &block = region.getBlocks().front();
  auto terminator = cast<gccjit::ReturnOp>(block.getTerminator());
  auto value = terminator->getOperand(0);
  auto expr = visitExpr(value, true);

  if (auto globalOp = dyn_cast<gccjit::GlobalOp>(parent)) {
    auto symName = SymbolRefAttr::get(getMLIRContext(), globalOp.getSymName());
    auto *lvalue = getTranslator().getGlobalLValue(symName);
    gcc_jit_global_set_initializer_rvalue(lvalue, expr);
    return {};
  }

  if (auto exprOp = dyn_cast<ExprOp>(parent))
    return expr;

  llvm_unreachable("unknown region parent");
}

Expr RegionVisitor::visitExpr(Value value, bool toplevel) {
  auto *cached = toplevel ? nullptr : &lookupExpr(value);

  if (toplevel || !*cached) {
    auto *op = value.getDefiningOp();
    if (op->getNumResults() != 1)
      llvm_unreachable("expected single result operation");

    Expr res =
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
            .Case([&](ExprOp op) { return visitExprWithoutCache(op); })
            .Case([&](DerefOp op) { return visitExprWithoutCache(op); })
            .Case([&](AccessFieldOp op) { return visitExprWithoutCache(op); })
            .Case([&](NewStructOp op) { return visitExprWithoutCache(op); })
            .Case([&](NewArrayOp op) { return visitExprWithoutCache(op); })
            .Case([&](NewUnionOp op) { return visitExprWithoutCache(op); })
            .Case([&](NewVectorOp op) { return visitExprWithoutCache(op); })
            .Case([&](DerefFieldOp op) { return visitExprWithoutCache(op); })
            .Case([&](AtomicLoadOp op) { return visitExprWithoutCache(op); })
            .Case([&](AtomicRMWOp op) { return visitExprWithoutCache(op); })
            .Case([&](AtomicCompareExchangeOp op) {
              return visitExprWithoutCache(op);
            })
            .Default([](Operation *op) -> Expr {
              op->dump();
              llvm::report_fatal_error("unknown expression type");
            });

    if (!toplevel)
      *cached = res;
    return res;
  }
  return *cached;
}

gcc_jit_lvalue *RegionVisitor::visitExprWithoutCache(DerefOp op) {
  auto ptr = visitExpr(op.getPtr());
  auto *loc = getTranslator().getLocation(op.getLoc());
  if (!op.getOffset())
    return gcc_jit_rvalue_dereference(ptr, loc);
  auto offset = visitExpr(op.getOffset());
  return gcc_jit_context_new_array_access(getContext(), loc, ptr, offset);
}

Expr RegionVisitor::visitExprWithoutCache(AccessFieldOp op) {
  auto composite = visitExpr(op.getComposite());
  auto *loc = getTranslator().getLocation(op.getLoc());
  auto ty = op.getComposite().getType();
  if (auto lvTy = dyn_cast<LValueType>(ty))
    ty = lvTy.getInnerType();
  auto compositeTy = getTranslator().getOrCreateRecordEntry(ty);
  auto index = op.getField().getZExtValue();
  auto *field = compositeTy[index];
  if (isa<LValueType>(op.getType()))
    return gcc_jit_lvalue_access_field(composite, loc, field);
  return gcc_jit_rvalue_access_field(composite, loc, field);
}

gcc_jit_lvalue *RegionVisitor::visitExprWithoutCache(DerefFieldOp op) {
  auto ptr = visitExpr(op.getPtr());
  auto *loc = getTranslator().getLocation(op.getLoc());
  auto ty = op.getPtr().getType();
  auto pointee = ty.getElementType();
  auto compositeTy = getTranslator().getOrCreateRecordEntry(pointee);
  auto index = op.getField().getZExtValue();
  auto *field = compositeTy[index];
  return gcc_jit_rvalue_dereference_field(ptr, loc, field);
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(NewStructOp op) {
  auto record = getTranslator().getOrCreateRecordEntry(op.getType());
  if (!record.isStruct())
    llvm_unreachable("expected struct type");
  llvm::SmallVector<gcc_jit_field *> fields;
  llvm::SmallVector<gcc_jit_rvalue *> values;
  for (auto field : op.getIndices())
    fields.push_back(record[field]);
  visitExprAsRValue(op.getElements(), values);
  auto *loc = getTranslator().getLocation(op.getLoc());
  return gcc_jit_context_new_struct_constructor(
      getContext(), loc, record.getAsType(), values.size(), fields.data(),
      values.data());
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(NewUnionOp op) {
  auto record = getTranslator().getOrCreateRecordEntry(op.getType());
  if (!record.isUnion())
    llvm_unreachable("expected union type");
  auto *field = record[op.getIndex().getZExtValue()];
  auto *loc = getTranslator().getLocation(op.getLoc());
  auto value = visitExpr(op.getElement());
  return gcc_jit_context_new_union_constructor(
      getContext(), loc, record.getAsType(), field, value);
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(NewVectorOp op) {
  auto *vectorTy = getTranslator().convertType(op.getType());
  auto *loc = getTranslator().getLocation(op.getLoc());
  llvm::SmallVector<gcc_jit_rvalue *> values;
  visitExprAsRValue(op.getElements(), values);
  return gcc_jit_context_new_rvalue_from_vector(getContext(), loc, vectorTy,
                                                values.size(), values.data());
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(NewArrayOp op) {
  auto *arrayTy = getTranslator().convertType(op.getType());
  auto *loc = getTranslator().getLocation(op.getLoc());
  llvm::SmallVector<gcc_jit_rvalue *> values;
  visitExprAsRValue(op.getElements(), values);
  return gcc_jit_context_new_array_constructor(getContext(), loc, arrayTy,
                                               values.size(), values.data());
}

static gcc_jit_rvalue *
buildBuiltinFuncCall(GCCJITTranslation &trans, const char *funcName,
                     gcc_jit_location *loc,
                     llvm::ArrayRef<gcc_jit_rvalue *> args) {
  gcc_jit_function *func =
      gcc_jit_context_get_builtin_function(trans.getContext(), funcName);
  return gcc_jit_context_new_call(trans.getContext(), loc, func, args.size(),
                                  const_cast<gcc_jit_rvalue **>(args.data()));
}

static gcc_jit_rvalue *translateAtomicOrdering(GCCJITTranslation &trans,
                                               AtomicOrdering ordering) {
  int gccMemOrder;
  switch (ordering) {
  case AtomicOrdering::Relaxed:
    gccMemOrder = __ATOMIC_RELAXED;
    break;
  case AtomicOrdering::Consume:
    gccMemOrder = __ATOMIC_CONSUME;
    break;
  case AtomicOrdering::Acquire:
    gccMemOrder = __ATOMIC_ACQUIRE;
    break;
  case AtomicOrdering::Release:
    gccMemOrder = __ATOMIC_RELEASE;
    break;
  case AtomicOrdering::AcqRel:
    gccMemOrder = __ATOMIC_ACQ_REL;
    break;
  case AtomicOrdering::SeqCst:
    gccMemOrder = __ATOMIC_SEQ_CST;
    break;
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#endif
  default:
    llvm_unreachable("unknown atomic ordering");
#ifdef __clang__
#pragma clang diagnostic pop
#endif
  }

  gcc_jit_type *gccMemOrderType =
      gcc_jit_context_get_type(trans.getContext(), GCC_JIT_TYPE_INT);
  return gcc_jit_context_new_rvalue_from_int(trans.getContext(),
                                             gccMemOrderType, gccMemOrder);
}

static gcc_jit_rvalue *castToCVVoidPtr(GCCJITTranslation &trans,
                                       gcc_jit_rvalue *value,
                                       gcc_jit_location *loc) {
  gcc_jit_type *cvVoidPtrTy =
      gcc_jit_type_get_pointer(gcc_jit_type_get_volatile(gcc_jit_type_get_const(
          gcc_jit_context_get_type(trans.getContext(), GCC_JIT_TYPE_VOID))));
  return gcc_jit_context_new_cast(trans.getContext(), loc, value, cvVoidPtrTy);
}

static gcc_jit_rvalue *castToVolatileVoidPtr(GCCJITTranslation &trans,
                                             gcc_jit_rvalue *value,
                                             gcc_jit_location *loc) {
  gcc_jit_type *cvVoidPtrTy =
      gcc_jit_type_get_pointer(gcc_jit_type_get_volatile(
          gcc_jit_context_get_type(trans.getContext(), GCC_JIT_TYPE_VOID)));
  return gcc_jit_context_new_cast(trans.getContext(), loc, value, cvVoidPtrTy);
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(AtomicLoadOp op) {
  gcc_jit_location *loc = translator.getLocation(op.getLoc());

  gcc_jit_rvalue *ptr =
      castToCVVoidPtr(translator, visitExpr(op.getPtr()), loc);
  gcc_jit_rvalue *order = translateAtomicOrdering(translator, op.getOrdering());

  size_t objectSize =
      translator.getTypeSize(op.getPtr().getType().getElementType());
  std::string builtinName = "__atomic_load_" + std::to_string(objectSize);
  gcc_jit_rvalue *result =
      buildBuiltinFuncCall(translator, builtinName.c_str(), loc, {ptr, order});

  if (mlir::isa<IntegerType, IntType>(op.getType()))
    return result;

  return gcc_jit_context_new_bitcast(translator.getContext(), loc, result,
                                     translator.convertType(op.getType()));
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(AtomicRMWOp op) {
  gcc_jit_location *loc = translator.getLocation(op.getLoc());

  gcc_jit_rvalue *ptr =
      castToVolatileVoidPtr(translator, visitExpr(op.getPtr()), loc);
  gcc_jit_rvalue *val = visitExpr(op.getValue());
  gcc_jit_rvalue *order = translateAtomicOrdering(translator, op.getOrdering());

  size_t objectSize =
      translator.getTypeSize(op.getPtr().getType().getElementType());
  if (!mlir::isa<IntegerType, IntType>(op.getValue().getType())) {
    gcc_jit_type *intTy =
        gcc_jit_context_get_int_type(translator.getContext(), objectSize, true);
    val = gcc_jit_context_new_bitcast(translator.getContext(), loc, val, intTy);
  }

  std::string builtinFuncName;
  switch (op.getKind()) {
  case AtomicRMWKind::Xchg:
    builtinFuncName = "__atomic_exchange_";
    break;
  case AtomicRMWKind::AddFetch:
    builtinFuncName = "__atomic_add_fetch_";
    break;
  case AtomicRMWKind::SubFetch:
    builtinFuncName = "__atomic_sub_fetch_";
    break;
  case AtomicRMWKind::AndFetch:
    builtinFuncName = "__atomic_and_fetch_";
    break;
  case AtomicRMWKind::NandFetch:
    builtinFuncName = "__atomic_nand_fetch_";
    break;
  case AtomicRMWKind::OrFetch:
    builtinFuncName = "__atomic_or_fetch_";
    break;
  case AtomicRMWKind::FetchAdd:
    builtinFuncName = "__atomic_fetch_add_";
    break;
  case AtomicRMWKind::FetchSub:
    builtinFuncName = "__atomic_fetch_sub_";
    break;
  case AtomicRMWKind::FetchAnd:
    builtinFuncName = "__atomic_fetch_and_";
    break;
  case AtomicRMWKind::FetchNand:
    builtinFuncName = "__atomic_fetch_nand_";
    break;
  case AtomicRMWKind::FetchOr:
    builtinFuncName = "__atomic_fetch_or_";
    break;
  case AtomicRMWKind::FetchXor:
    builtinFuncName = "__atomic_fetch_xor_";
    break;
  default:
    llvm_unreachable("unknown atomic rmw op kind");
  }

  builtinFuncName += std::to_string(objectSize);
  gcc_jit_rvalue *result = buildBuiltinFuncCall(
      translator, builtinFuncName.c_str(), loc, {ptr, val, order});

  if (mlir::isa<IntegerType, IntType>(op.getType()))
    return result;
  return gcc_jit_context_new_bitcast(translator.getContext(), loc, result,
                                     translator.convertType(op.getType()));
}

gcc_jit_rvalue *
RegionVisitor::visitExprWithoutCache(AtomicCompareExchangeOp op) {
  gcc_jit_location *loc = translator.getLocation(op.getLoc());

  gcc_jit_rvalue *ptr =
      castToVolatileVoidPtr(translator, visitExpr(op.getPtr()), loc);
  gcc_jit_rvalue *expected =
      castToCVVoidPtr(translator, visitExpr(op.getExpected()), loc);
  gcc_jit_rvalue *desired = visitExpr(op.getDesired());
  gcc_jit_rvalue *successOrdering =
      translateAtomicOrdering(translator, op.getSuccessOrd());
  gcc_jit_rvalue *failureOrdering =
      translateAtomicOrdering(translator, op.getFailureOrd());

  size_t objectSize =
      translator.getTypeSize(op.getPtr().getType().getElementType());
  if (!mlir::isa<IntegerType, IntType>(op.getDesired().getType())) {
    gcc_jit_type *intTy =
        gcc_jit_context_get_int_type(translator.getContext(), objectSize, true);
    desired = gcc_jit_context_new_bitcast(translator.getContext(), loc, desired,
                                          intTy);
  }

  gcc_jit_type *boolTy =
      gcc_jit_context_get_type(translator.getContext(), GCC_JIT_TYPE_BOOL);
  gcc_jit_rvalue *weak =
      op.getWeak() ? gcc_jit_context_one(translator.getContext(), boolTy)
                   : gcc_jit_context_zero(translator.getContext(), boolTy);

  std::string builtinFuncName =
      "__atomic_compare_exchange_" + std::to_string(objectSize);
  return buildBuiltinFuncCall(
      translator, builtinFuncName.c_str(), loc,
      {ptr, expected, desired, weak, successOrdering, failureOrdering});
}

Expr RegionVisitor::visitExprWithoutCache(ExprOp op) {
  RegionVisitor visitor(getTranslator(), op.getRegion(), this);
  return visitor.translateIntoContext();
}

void RegionVisitor::visitConditionalOp(gcc_jit_block *blk, ConditionalOp op) {
  auto condition = visitExpr(op.getCondition());
  auto *trueBlock = lookupBlock(op.getOnTrue());
  auto *falseBlock = lookupBlock(op.getOnFalse());
  auto *loc = getTranslator().getLocation(op.getLoc());
  gcc_jit_block_end_with_conditional(blk, loc, condition, trueBlock,
                                     falseBlock);
}

void RegionVisitor::visitAtomicStoreOp(gcc_jit_block *blk, AtomicStoreOp op) {
  gcc_jit_location *loc = translator.getLocation(op.getLoc());

  gcc_jit_rvalue *ptr =
      castToVolatileVoidPtr(translator, visitExpr(op.getPtr()), loc);
  gcc_jit_rvalue *value = visitExpr(op.getValue());
  gcc_jit_rvalue *order = translateAtomicOrdering(translator, op.getOrdering());

  size_t objectSize =
      translator.getTypeSize(op.getPtr().getType().getElementType());
  if (!mlir::isa<IntegerType, IntType>(op.getValue().getType())) {
    gcc_jit_type *intTy =
        gcc_jit_context_get_int_type(translator.getContext(), objectSize, true);
    value =
        gcc_jit_context_new_bitcast(translator.getContext(), loc, value, intTy);
  }

  std::string builtinFuncName = "__atomic_store_" + std::to_string(objectSize);
  gcc_jit_rvalue *builtinCall = buildBuiltinFuncCall(
      translator, builtinFuncName.c_str(), loc, {ptr, value, order});

  gcc_jit_block_add_eval(blk, loc, builtinCall);
}

void RegionVisitor::visitExprAsRValue(
    ValueRange operands, llvm::SmallVectorImpl<gcc_jit_rvalue *> &result) {
  for (auto operand : operands)
    result.push_back(visitExpr(operand));
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
  auto string = op.getValue().str();
  return gcc_jit_context_new_string_literal(getContext(), string.c_str());
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(SizeOfOp op) {
  auto type = op.getType();
  auto *typeHandle = getTranslator().convertType(type);
  auto *size = gcc_jit_context_new_sizeof(getContext(), typeHandle);
  auto *loc = getTranslator().getLocation(op.getLoc());
  auto resTy = op.getResult().getType();
  auto *resTyHandle = getTranslator().convertType(resTy);
  if (resTy.getKind() != GCC_JIT_TYPE_INT)
    size = gcc_jit_context_new_cast(getContext(), loc, size, resTyHandle);
  return size;
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(AlignOfOp op) {
#ifdef LIBGCCJIT_HAVE_gcc_jit_context_new_alignof
  auto type = op.getType();
  auto *typeHandle = getTranslator().convertType(type);
  auto *align = gcc_jit_context_new_alignof(getContext(), typeHandle);
  auto *loc = getTranslator().getLocation(op.getLoc());
  auto resTy = op.getResult().getType();
  auto *resTyHandle = getTranslator().convertType(resTy);
  if (resTy.getKind() != GCC_JIT_TYPE_INT)
    align = gcc_jit_context_new_cast(getContext(), loc, align, resTyHandle);
  return align;
#endif
  auto type = op.getType();
  auto *typeHandle = getTranslator().convertType(type);
  auto *resTyHandle = getTranslator().convertType(op.getResult().getType());
  auto *charTyHandle =
      gcc_jit_context_get_type(getContext(), GCC_JIT_TYPE_CHAR);
  auto *charField =
      gcc_jit_context_new_field(getContext(), nullptr, charTyHandle, "c");
  auto *typeField =
      gcc_jit_context_new_field(getContext(), nullptr, typeHandle, "t");
  gcc_jit_field *fields[] = {charField, typeField};
  auto *fakeStruct =
      gcc_jit_context_new_struct_type(getContext(), nullptr, "fake", 2, fields);
  auto *fakeStructHandle = gcc_jit_struct_as_type(fakeStruct);
  auto *fakeStructPtrHandle = gcc_jit_type_get_pointer(fakeStructHandle);
  auto *nullStructPtr = gcc_jit_context_null(getContext(), fakeStructPtrHandle);
  auto *fakeStructLvalue = gcc_jit_rvalue_dereference(nullStructPtr, nullptr);
  auto *fakeFieldAccess =
      gcc_jit_lvalue_access_field(fakeStructLvalue, nullptr, typeField);
  auto *fakeFieldAddr = gcc_jit_lvalue_get_address(fakeFieldAccess, nullptr);
  auto *indexTy = gcc_jit_context_get_type(getContext(), GCC_JIT_TYPE_SIZE_T);
  auto *fakeFieldAddrValue = gcc_jit_context_new_bitcast(
      getContext(), nullptr, fakeFieldAddr, indexTy);
  if (indexTy != resTyHandle)
    return gcc_jit_context_new_cast(getContext(), nullptr, fakeFieldAddrValue,
                                    resTyHandle);
  return fakeFieldAddrValue;
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(AsRValueOp op) {
  auto lvalue = visitExpr(op.getLvalue());
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
  auto lhs = visitExpr(op.getLhs());
  auto rhs = visitExpr(op.getRhs());
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
  auto operand = visitExpr(op.getOperand());
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
  auto lhs = visitExpr(op.getLhs());
  auto rhs = visitExpr(op.getRhs());
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
  if (op.getTail())
    gcc_jit_rvalue_set_bool_require_tail_call(call, true);
  return call;
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(CastOp op) {
  auto operand = visitExpr(op.getOperand());
  auto *loc = getTranslator().getLocation(op.getLoc());
  auto *ctxt = getContext();
  auto *type = getTranslator().convertType(op.getType());
  return gcc_jit_context_new_cast(ctxt, loc, operand, type);
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(BitCastOp op) {
  auto operand = visitExpr(op.getOperand());
  auto *loc = getTranslator().getLocation(op.getLoc());
  auto *ctxt = getContext();
  auto *type = getTranslator().convertType(op.getType());
  return gcc_jit_context_new_bitcast(ctxt, loc, operand, type);
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(PtrCallOp op) {
  auto callee = visitExpr(op.getCallee());
  llvm::SmallVector<gcc_jit_rvalue *> args;
  visitExprAsRValue(op.getArgs(), args);
  auto *loc = getTranslator().getLocation(op.getLoc());
  auto *ctxt = getContext();
  auto *call = gcc_jit_context_new_call_through_ptr(ctxt, loc, callee,
                                                    args.size(), args.data());
  if (op.getTail())
    gcc_jit_rvalue_set_bool_require_tail_call(call, true);
  return call;
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(AddrOp op) {
  auto lvalue = visitExpr(op.getOperand());
  auto *loc = getTranslator().getLocation(op.getLoc());
  return gcc_jit_lvalue_get_address(lvalue, loc);
}

gcc_jit_rvalue *RegionVisitor::visitExprWithoutCache(FnAddrOp op) {
  auto fn = getTranslator().getFunction(op.getCallee());
  assert(fn && "function not found");
  auto *loc = getTranslator().getLocation(op.getLoc());
  return gcc_jit_function_get_address(fn, loc);
}

gcc_jit_lvalue *RegionVisitor::visitExprWithoutCache(GetGlobalOp op) {
  auto *lvalue = getTranslator().getGlobalLValue(op.getSym());
  assert(lvalue && "global not found");
  return lvalue;
}

void RegionVisitor::visitAssignOp(gcc_jit_block *blk, AssignOp op) {
  auto lvalue = visitExpr(op.getLvalue());
  auto rvalue = visitExpr(op.getRvalue());
  auto *loc = getTranslator().getLocation(op.getLoc());
  gcc_jit_block_add_assignment(blk, loc, lvalue, rvalue);
}

void RegionVisitor::visitUpdateOp(gcc_jit_block *blk, UpdateOp op) {
  auto lvalue = visitExpr(op.getLvalue());
  auto rvalue = visitExpr(op.getRvalue());
  auto *loc = getTranslator().getLocation(op.getLoc());
  auto kind = convertBinaryOp(op.getOp());
  gcc_jit_block_add_assignment_op(blk, loc, lvalue, kind, rvalue);
}

void RegionVisitor::visitReturnOp(gcc_jit_block *blk, ReturnOp op) {
  if (op->getNumOperands())
    gcc_jit_block_end_with_return(blk, getTranslator().getLocation(op.getLoc()),
                                  visitExpr(op.getOperand(0)));
  else
    gcc_jit_block_end_with_void_return(
        blk, getTranslator().getLocation(op.getLoc()));
}

gcc_jit_block *RegionVisitor::lookupBlock(Block *block) {
  auto it = blocks.find(block);
  if (it == blocks.end() && parent)
    return parent->lookupBlock(block);
  return it->second;
}

Expr *RegionVisitor::lookupExprImpl(Value value) {
  if (parent) {
    auto *expr = parent->lookupExprImpl(value);
    if (expr)
      return expr;
  }
  auto it = exprCache.find(value);
  if (it != exprCache.end())
    return &it->second;
  return nullptr;
}

Expr &RegionVisitor::lookupExpr(Value value) {
  if (auto *expr = lookupExprImpl(value))
    return *expr;
  return exprCache[value];
}

void RegionVisitor::visitSwitchOp(gcc_jit_block *blk, SwitchOp op) {
  auto value = visitExpr(op.getValue());
  auto *loc = getTranslator().getLocation(op.getLoc());
  llvm::SmallVector<gcc_jit_case *> cases;
  for (auto [lb, ub, dst] :
       llvm::zip(op.getCaseLowerbound(), op.getCaseUpperbound(),
                 op.getCaseDestinations())) {
    // TODO: handle signedness
    auto lbAttr = cast<IntAttr>(lb);
    auto ubAttr = cast<IntAttr>(ub);
    auto intLb = lbAttr.getValue().getZExtValue();
    auto intUb = ubAttr.getValue().getZExtValue();
    auto *dstBlk = blocks[dst];
    auto *lbv = gcc_jit_context_new_rvalue_from_long(
        getContext(), getTranslator().convertType(lbAttr.getType()), intLb);
    auto *ubv = gcc_jit_context_new_rvalue_from_long(
        getContext(), getTranslator().convertType(ubAttr.getType()), intUb);
    cases.push_back(gcc_jit_context_new_case(getContext(), lbv, ubv, dstBlk));
  }
  auto *defaultBlk = blocks[op.getDefaultDestination()];
  gcc_jit_block_end_with_switch(blk, loc, value, defaultBlk, cases.size(),
                                cases.data());
}

void RegionVisitor::visitJumpOp(gcc_jit_block *blk, JumpOp op) {
  auto *dst = blocks[op.getDest()];
  gcc_jit_block_end_with_jump(blk, getTranslator().getLocation(op.getLoc()),
                              dst);
}

template <typename OpTy>
void RegionVisitor::populateExtendedAsm(gcc_jit_extended_asm *extAsm, OpTy op) {
  auto asmStr = op.getTemplateCode().str();
  for (auto [output, constraint, symbol] : llvm::zip(
           op.getOutputs(), op.getOutputConstraints(), op.getOutputSymbols())) {
    auto constraintStr = cast<StringAttr>(constraint).getValue().str();
    auto symbolStr = cast<StringAttr>(symbol).getValue().str();
    auto lvalue = visitExpr(output);
    assert(lvalue.isLValue() && "expected lvalue");
    gcc_jit_extended_asm_add_output_operand(
        extAsm, symbolStr.empty() ? nullptr : symbolStr.c_str(),
        constraintStr.c_str(), lvalue);
  }
  for (auto [input, constraint, symbol] : llvm::zip(
           op.getInputs(), op.getInputConstraints(), op.getInputSymbols())) {
    auto constraintStr = cast<StringAttr>(constraint).getValue().str();
    auto symbolStr = cast<StringAttr>(symbol).getValue().str();
    auto rvalue = visitExpr(input);
    gcc_jit_extended_asm_add_input_operand(
        extAsm, symbolStr.empty() ? nullptr : symbolStr.c_str(),
        constraintStr.c_str(), rvalue);
  }
  for (auto clobber : op.getClobbers()) {
    auto clobberStr = cast<StringAttr>(clobber).getValue().str();
    gcc_jit_extended_asm_add_clobber(extAsm, clobberStr.c_str());
  }
}

void RegionVisitor::visitAsmOp(gcc_jit_block *blk, AsmOp op) {
  auto *loc = getTranslator().getLocation(op.getLoc());
  auto asmStr = op.getTemplateCode().str();
  auto *extendedAsm = gcc_jit_block_add_extended_asm(blk, loc, asmStr.c_str());
  populateExtendedAsm(extendedAsm, op);
}

void RegionVisitor::visitAsmGotoOp(gcc_jit_block *blk, AsmGotoOp op) {
  auto *loc = getTranslator().getLocation(op.getLoc());
  auto asmStr = op.getTemplateCode().str();
  llvm::SmallVector<gcc_jit_block *> targets;
  for (auto *target : op.getLabels())
    targets.push_back(blocks.at(target));
  gcc_jit_block *fallthrough = blocks.at(op.getFallthrough());
  auto *extendedAsm = gcc_jit_block_end_with_extended_asm_goto(
      blk, loc, asmStr.c_str(), targets.size(), targets.data(), fallthrough);
  populateExtendedAsm(extendedAsm, op);
}

void GCCJITTranslation::translateFunctions() {
  for (auto func : moduleOp.getOps<gccjit::FuncOp>()) {
    auto &region = func.getBody();
    RegionVisitor visitor(*this, region);
    visitor.translateIntoContext();
  }
}

//===----------------------------------------------------------------------===//
// TranslateModuleToGCCJIT
//===----------------------------------------------------------------------===//
llvm::Expected<GCCJITContext> translateModuleToGCCJIT(ModuleOp op) {
  GCCJITTranslation translator;
  translator.translateModuleToGCCJIT(op);
  auto ctx = translator.takeContext();
  if (const auto *err = gcc_jit_context_get_last_error(ctx.get()))
    return llvm::createStringError(llvm::inconvertibleErrorCode(), err);
  return ctx;
}

///===----------------------------------------------------------------------===///
/// GCCJITContextDeleter
///===----------------------------------------------------------------------===///
void GCCJITContextDeleter::operator()(gcc_jit_context *ctxt) const {
  if (ctxt)
    gcc_jit_context_release(ctxt);
}

GCCJITTranslation::GCCJITRecord
GCCJITTranslation::getOrCreateRecordEntry(Type type) {
  return llvm::TypeSwitch<Type, GCCJITRecord>(type)
      .Case([&](StructType t) { return &getOrCreateStructEntry(t); })
      .Case([&](UnionType t) { return &getOrCreateUnionEntry(t); })
      .Default([&](Type) -> GCCJITRecord {
        type.dump();
        llvm_unreachable("unexpected type");
      });
}
} // namespace mlir::gccjit
