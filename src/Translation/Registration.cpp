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

namespace mlir::gccjit {
namespace {
llvm::Expected<llvm::sys::fs::TempFile>
dumpContextToTempfile(gcc_jit_context *ctxt, bool reproducer) {
  auto file = llvm::sys::fs::TempFile::create("mlir-gccjit-%%%%%%%");
  if (!file)
    return file.takeError();
  if (reproducer)
    gcc_jit_context_dump_reproducer_to_file(ctxt, file->TmpName.c_str());
  else
    gcc_jit_context_dump_to_file(ctxt, file->TmpName.c_str(), false);
  return file;
}

LogicalResult copyFileToStream(llvm::sys::fs::TempFile file,
                               llvm::raw_ostream &os) {
  os.flush();
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(file.TmpName);
  if (!buffer)
    return failure();
  os << buffer.get()->getBuffer();
  return success();
}

void registerTranslation(llvm::StringRef name, llvm::StringRef desc,
                         bool reproducer) {
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
      [](DialectRegistry &registry) {
        registry.insert<gccjit::GCCJITDialect>();
      });
}

} // namespace

void registerToGCCJITGimpleTranslation() {
  registerTranslation("mlir-to-gccjit-gimple",
                      "Translate MLIR to GCCJIT's GIMPLE format", false);
}

void registerToGCCJITReproducerTranslation() {
  registerTranslation("mlir-to-gccjit-reproducer",
                      "Translate MLIR to GCCJIT's reproducer format", true);
}
} // namespace mlir::gccjit
