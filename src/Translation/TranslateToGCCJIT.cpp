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
#include "mlir-gccjit/IR/GCCJITDialect.h"
#include "mlir-gccjit/IR/GCCJITOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MemoryBuffer.h"
namespace mlir::gccjit {

namespace {
void populateGCCJITModuleOptions(gcc_jit_context *context, ModuleOp op) {
  for (auto &attr : op->getAttrs()) {
    if (attr.getName() == "gccjit.prog_name") {
      if (auto strAttr = dyn_cast<StringAttr>(attr.getValue()))
        gcc_jit_context_set_str_option(context, GCC_JIT_STR_OPTION_PROGNAME, strAttr.str().c_str());
    } else if (attr.getName() == "gccjit.opt_level") {
      if (auto intAttr = dyn_cast<OptLevelAttr>(attr.getValue())) {
        int optLevel = static_cast<int>(intAttr.getLevel().getValue());
        gcc_jit_context_set_int_option(context, GCC_JIT_INT_OPTION_OPTIMIZATION_LEVEL, optLevel);
      }
    } else if (attr.getName() == "gccjit.allow_unreachable") {
      if (auto boolAttr = dyn_cast<BoolAttr>(attr.getValue()))
        gcc_jit_context_set_bool_allow_unreachable_blocks(context, boolAttr.getValue());
    }
  }
}

llvm::Expected<llvm::sys::fs::TempFile> dumpContextToTempfile(gcc_jit_context *ctxt,
                                                              bool reproducer) {
  auto file = llvm::sys::fs::TempFile::create("mlir-gccjit-%%%%%%%");
  if (!file)
    return file.takeError();
  if (reproducer)
    gcc_jit_context_dump_reproducer_to_file(ctxt, file->TmpName.c_str());
  else
    gcc_jit_context_dump_to_file(ctxt, file->TmpName.c_str(), false);
  return file;
}

llvm::LogicalResult copyFileToStream(llvm::sys::fs::TempFile file, llvm::raw_ostream &os) {
  os.flush();
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(file.TmpName);
  if (!buffer)
    return mlir::failure();
  os << buffer.get()->getBuffer();
  return mlir::success();
}

void registerTranslation(llvm::StringRef name, llvm::StringRef desc, bool reproducer) {
  TranslateFromMLIRRegistration registration(
      name, desc,
      [reproducer](Operation *op, raw_ostream &output) {
        auto module = dyn_cast<ModuleOp>(op);
        if (!module) {
          op->emitError("expected 'module' operation");
          return failure();
        }
        auto context = translateModuleToGCCJIT(module);
        if (!context) {
          op->emitError("failed to translate to GCCJIT context");
          return failure();
        }
        auto file = dumpContextToTempfile(context.get().get(), reproducer);
        if (!file) {
          op->emitError("failed to dump GCCJIT context to tempfile");
          return failure();
        }
        return copyFileToStream(std::move(*file), output);
      },
      [](DialectRegistry &registry) { registry.insert<gccjit::GCCJITDialect>(); });
}

class GCCJITTypeConverter {
  llvm::DenseMap<mlir::Type, gcc_jit_type *> typeMap;
  gcc_jit_context *ctxt;

public:
  GCCJITTypeConverter(gcc_jit_context *ctxt) : ctxt(ctxt) {}
  void convertTypes(mlir::TypeRange types, llvm::SmallVectorImpl<gcc_jit_type *> &result) {
    for (auto type : types)
      result.push_back(convertType(type));
  }
  gcc_jit_type *convertType(mlir::Type type) {
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
                    .Default([](mlir::Type) { return nullptr; });
    typeMap[type] = res;
    return res;
  }
};

gcc_jit_function_kind convertFnKind(FnKind kind) {
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

void declareAllFunctionAndGlobals(gcc_jit_context *ctxt, ModuleOp op) {
  // todo: declare globals
  // this should be wrapped in a global context holder
  GCCJITTypeConverter typeConverter(ctxt);
  op.walk([&](gccjit::FuncOp func) {
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
    auto *funcType = gcc_jit_context_new_function(ctxt, /*todo: location*/ nullptr, kind,
                                                  returnType, name.c_str(), paramTypes.size(),
                                                  params.data(), type.isVarArg());
    (void)funcType;
  });
}

} // namespace

llvm::Expected<GCCJITContext> translateModuleToGCCJIT(ModuleOp op) {
  gcc_jit_context *ctxt = gcc_jit_context_acquire();
  populateGCCJITModuleOptions(ctxt, op);
  declareAllFunctionAndGlobals(ctxt, op);
  return GCCJITContext(ctxt);
}

void registerToGCCJITGimpleTranslation() {
  registerTranslation("mlir-to-gccjit-gimple", "Translate MLIR to GCCJIT's GIMPLE format", false);
}

void registerToGCCJITReproducerTranslation() {
  registerTranslation("mlir-to-gccjit-reproducer", "Translate MLIR to GCCJIT's reproducer format",
                      true);
}

} // namespace mlir::gccjit
