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
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
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

GCCJITTypeConverter::GCCJITTypeConverter(std::unique_ptr<impl::GCCJITTypeConverter> impl)
    : impl(std::move(impl)) {}

void GCCJITTypeConverter::convertTypes(mlir::TypeRange types,
                                       llvm::SmallVectorImpl<gcc_jit_type *> &result) {
  for (auto type : types)
    result.push_back(convertType(type));
}
gcc_jit_type *GCCJITTypeConverter::convertType(mlir::Type type) {
  auto &typeMap = impl->typeMap;
  if (auto it = typeMap.find(type); it != typeMap.end())
    return it->second;
  auto *res = llvm::TypeSwitch<mlir::Type, gcc_jit_type *>(type)
                  .Case([&](gccjit::LValueType t) { return convertType(t.getInnerType()); })
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
                    return gcc_jit_context_get_type(impl->getContext(), GCC_JIT_TYPE_VOID);
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
    : ctxt(gcc_jit_context_acquire()), typeConverter(std::make_unique<GCCJITTypeConverter>(*this)) {
}
Translator::~Translator() {
  if (ctxt) {
    gcc_jit_context_release(ctxt);
  }
}
void Translator::populateGCCJITModuleOptions() {
  for (auto &attr : moduleOp->getAttrs()) {
    if (attr.getName() == "gccjit.prog_name") {
      if (auto strAttr = dyn_cast<StringAttr>(attr.getValue()))
        gcc_jit_context_set_str_option(ctxt, GCC_JIT_STR_OPTION_PROGNAME, strAttr.str().c_str());
    } else if (attr.getName() == "gccjit.opt_level") {
      if (auto intAttr = dyn_cast<OptLevelAttr>(attr.getValue())) {
        int optLevel = static_cast<int>(intAttr.getLevel().getValue());
        gcc_jit_context_set_int_option(ctxt, GCC_JIT_INT_OPTION_OPTIMIZATION_LEVEL, optLevel);
      }
    } else if (attr.getName() == "gccjit.allow_unreachable") {
      if (auto boolAttr = dyn_cast<BoolAttr>(attr.getValue()))
        gcc_jit_context_set_bool_allow_unreachable_blocks(ctxt, boolAttr.getValue());
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
    std::transform(
        enumerated.begin(), enumerated.end(), std::back_inserter(params), [&](auto pair) {
          auto index = pair.index();
          auto type = pair.value();
          auto name = llvm::Twine("arg").concat(llvm::Twine(index)).str();
          return gcc_jit_context_new_param(ctxt, /*todo: location*/ nullptr, type, name.c_str());
        });
    auto *funcHandle = gcc_jit_context_new_function(ctxt, /*todo: location*/ nullptr, kind,
                                                    returnType, name.c_str(), paramTypes.size(),
                                                    params.data(), type.isVarArg());
    SymbolRefAttr symRef = SymbolRefAttr::get(getMLIRContext(), name);
    functionMap[symRef] = {funcHandle, std::move(params)};
  });
}

MLIRContext *Translator::getMLIRContext() { return moduleOp.getContext(); }

} // namespace impl

GCCJITContext Translator::takeContext() {
  auto *ctxt = impl->ctxt;
  impl->ctxt = nullptr;
  return GCCJITContext(ctxt);
}
gcc_jit_context *Translator::getContext() const { return impl->ctxt; }
::mlir::gccjit::GCCJITTypeConverter &Translator::getTypeConverter() { return impl->typeConverter; }
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
