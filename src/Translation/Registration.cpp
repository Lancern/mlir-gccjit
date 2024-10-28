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

#include "libgccjit.h"
#include "mlir-gccjit/IR/GCCJITDialect.h"
#include "mlir-gccjit/Translation/TranslateToGCCJIT.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::gccjit {
namespace {
enum class OutputType {
  Gimple,
  Reproducer,
  Assembly,
  Object,
  Executable,
  Dylib
};

std::optional<llvm::SmallString<128>>
dumpContextToTempfile(gcc_jit_context *ctxt, OutputType type) {
  StringRef suffix;
  llvm::SmallString<128> path;
  switch (type) {
  case OutputType::Gimple:
    suffix = ".gimple";
    break;
  case OutputType::Reproducer:
    suffix = ".c";
    break;
  case OutputType::Assembly:
    suffix = ".s";
    break;
  case OutputType::Object:
    suffix = ".o";
    break;
  case OutputType::Executable:
    suffix = ".exe";
    break;
  case OutputType::Dylib:
    suffix = ".so";
    break;
  }
  auto err = llvm::sys::fs::createTemporaryFile("mlir-gccjit", suffix, path);
  if (err)
    return std::nullopt;
  switch (type) {
  case OutputType::Gimple:
    gcc_jit_context_dump_to_file(ctxt, path.c_str(), false);
    break;
  case OutputType::Reproducer:
    gcc_jit_context_dump_reproducer_to_file(ctxt, path.c_str());
    break;
  case OutputType::Assembly:
    gcc_jit_context_compile_to_file(ctxt, GCC_JIT_OUTPUT_KIND_ASSEMBLER,
                                    path.c_str());
    break;
  case OutputType::Object:
    gcc_jit_context_compile_to_file(ctxt, GCC_JIT_OUTPUT_KIND_OBJECT_FILE,
                                    path.c_str());
    break;
  case OutputType::Executable:
    gcc_jit_context_compile_to_file(ctxt, GCC_JIT_OUTPUT_KIND_EXECUTABLE,
                                    path.c_str());
    break;
  case OutputType::Dylib:
    gcc_jit_context_compile_to_file(ctxt, GCC_JIT_OUTPUT_KIND_DYNAMIC_LIBRARY,
                                    path.c_str());
    break;
  }
  return path;
}

LogicalResult copyFileToStream(const llvm::SmallString<128> &path,
                               llvm::raw_ostream &os) {
  os.flush();
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return failure();
  os << buffer.get()->getBuffer();
  return success();
}

void registerTranslation(llvm::StringRef name, llvm::StringRef desc,
                         OutputType type) {
  TranslateFromMLIRRegistration registration(
      name, desc,
      [type](Operation *op, raw_ostream &output) {
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
        auto file = dumpContextToTempfile(context.get().get(), type);
        if (!file) {
          op->emitError("failed to dump GCCJIT context to tempfile");
          return failure();
        }
        return copyFileToStream(std::move(*file), output);
      },
      [](DialectRegistry &registry) {
        registry.insert<gccjit::GCCJITDialect>();
      });
}

} // namespace

void registerToGCCJITGimpleTranslation() {
  registerTranslation("mlir-to-gccjit-gimple",
                      "Translate MLIR to GCCJIT's GIMPLE format",
                      OutputType::Gimple);
}

void registerToGCCJITReproducerTranslation() {
  registerTranslation("mlir-to-gccjit-reproducer",
                      "Translate MLIR to GCCJIT's reproducer format",
                      OutputType::Reproducer);
}

void registerToGCCJITAssemblyTranslation() {
  registerTranslation("mlir-to-gccjit-assembly",
                      "Translate MLIR to GCCJIT's assembly format",
                      OutputType::Assembly);
}

void registerToGCCJITObjectTranslation() {
  registerTranslation("mlir-to-gccjit-object",
                      "Translate MLIR to GCCJIT's object file format",
                      OutputType::Object);
}

void registerToGCCJITExecutableTranslation() {
  registerTranslation("mlir-to-gccjit-executable",
                      "Translate MLIR to GCCJIT's executable format",
                      OutputType::Executable);
}
void registerToGCCJITDylibTranslation() {
  registerTranslation("mlir-to-gccjit-dylib",
                      "Translate MLIR to GCCJIT's dynamic library format",
                      OutputType::Dylib);
}
} // namespace mlir::gccjit
