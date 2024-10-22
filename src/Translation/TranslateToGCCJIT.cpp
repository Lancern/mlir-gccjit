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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Error.h"
namespace mlir::gccjit {

//===----------------------------------------------------------------------===//
// Translator declaration
//===----------------------------------------------------------------------===//

namespace [[gnu::visibility("hidden")]] impl {

struct FunctionEntry {
  gcc_jit_function *fnHandle;
  llvm::SmallVector<gcc_jit_param *> params;
};

struct Translator {
  // Members
  gcc_jit_context *ctxt;
  ::mlir::gccjit::GCCJITTypeConverter typeConverter;
  llvm::DenseMap<mlir::SymbolRefAttr, FunctionEntry> functionMap;
  llvm::DenseMap<mlir::SymbolRefAttr, gcc_jit_lvalue *> globalMap;
  ModuleOp moduleOp;

  // Codegen status
  gcc_jit_function *currentFunction = nullptr;
  llvm::DenseMap<Block *, gcc_jit_block *> blockMap;
  llvm::DenseMap<Value, gcc_jit_lvalue *> localVarMap;
  llvm::DenseMap<Value, gcc_jit_rvalue *> valueMap;
  gcc_jit_block *currentBlock = nullptr;

  // APIs
  Translator();
  ~Translator();
  void populateGCCJITModuleOptions();
  void declareAllFunctionAndGlobals();
  static gcc_jit_function_kind convertFnKind(FnKind kind);
  MLIRContext *getMLIRContext();
  gcc_jit_location *getLocation(LocationAttr loc);
};

} // namespace impl

//===----------------------------------------------------------------------===//
// GCCJITTypeConverter
//===----------------------------------------------------------------------===//
namespace [[gnu::visibility("hidden")]] impl {
struct GCCJITTypeConverter {
  llvm::DenseMap<mlir::Type, gcc_jit_type *> typeMap;
  Translator &translator;
  GCCJITTypeConverter(Translator &translator) : translator(translator) {}
  gcc_jit_context *getContext() const { return translator.ctxt; }
};
} // namespace impl

GCCJITTypeConverter::GCCJITTypeConverter(
    std::unique_ptr<impl::GCCJITTypeConverter> impl)
    : impl(std::move(impl)) {}

void GCCJITTypeConverter::convertTypes(
    mlir::TypeRange types, llvm::SmallVectorImpl<gcc_jit_type *> &result) {
  for (auto type : types)
    result.push_back(convertType(type));
}

gcc_jit_type *GCCJITTypeConverter::convertType(mlir::Type type) {
  auto &typeMap = impl->typeMap;
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
                    return gcc_jit_context_get_type(impl->getContext(), kind);
                  })
                  .Case([&](gccjit::FloatType t) {
                    auto kind = t.getKind();
                    return gcc_jit_context_get_type(impl->getContext(), kind);
                  })
                  .Case([&](gccjit::ComplexType t) {
                    auto kind = t.getKind();
                    return gcc_jit_context_get_type(impl->getContext(), kind);
                  })
                  .Case([&](gccjit::VoidType t) {
                    return gcc_jit_context_get_type(impl->getContext(),
                                                    GCC_JIT_TYPE_VOID);
                  })
                  .Case([&](gccjit::ArrayType t) {
                    auto *elemType = convertType(t.getElementType());
                    auto size = t.getSize();
                    return gcc_jit_context_new_array_type(
                        impl->getContext(), nullptr, elemType, size);
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

//===----------------------------------------------------------------------===//
// Translator implementation
//===----------------------------------------------------------------------===//
namespace [[gnu::visibility("hidden")]] impl {

Translator::Translator()
    : ctxt(gcc_jit_context_acquire()),
      typeConverter(std::make_unique<GCCJITTypeConverter>(*this)) {}

Translator::~Translator() {
  if (ctxt) {
    gcc_jit_context_release(ctxt);
  }
}

void Translator::populateGCCJITModuleOptions() {
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

gcc_jit_function_kind Translator::convertFnKind(FnKind kind) {
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
}

void Translator::declareAllFunctionAndGlobals() {
  moduleOp.walk([&](gccjit::FuncOp func) {
    auto type = func.getFunctionType();
    llvm::SmallVector<gcc_jit_type *> paramTypes;
    llvm::SmallVector<gcc_jit_param *> params;
    typeConverter.convertTypes(type.getInputs(), paramTypes);
    auto *returnType = typeConverter.convertType(type.getReturnType());
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
    auto *typeHandle = typeConverter.convertType(type);
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
          .Case([&](StringInitializerAttr attr) {
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
    if (!global.getBody().empty()) {
      // TODO
    }
  });
}

MLIRContext *Translator::getMLIRContext() { return moduleOp.getContext(); }

gcc_jit_location *Translator::getLocation(LocationAttr loc) {
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

} // namespace impl

GCCJITContext Translator::takeContext() {
  auto *ctxt = impl->ctxt;
  impl->ctxt = nullptr;
  return GCCJITContext(ctxt);
}
gcc_jit_context *Translator::getContext() const { return impl->ctxt; }
::mlir::gccjit::GCCJITTypeConverter &Translator::getTypeConverter() {
  return impl->typeConverter;
}
void Translator::translateModuleToGCCJIT(ModuleOp op) {
  impl->moduleOp = op;
  impl->populateGCCJITModuleOptions();
  impl->declareAllFunctionAndGlobals();
}
Translator::Translator() : impl(std::make_unique<impl::Translator>()) {}

//===----------------------------------------------------------------------===//
// TranslateModuleToGCCJIT
//===----------------------------------------------------------------------===//

llvm::Expected<GCCJITContext> translateModuleToGCCJIT(ModuleOp op) {
  Translator translator;
  translator.translateModuleToGCCJIT(op);
  return translator.takeContext();
}

} // namespace mlir::gccjit
