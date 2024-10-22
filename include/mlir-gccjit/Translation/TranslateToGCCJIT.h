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

#ifndef MLIR_GCCJIT_TRANSLATION_TRANSLATETOGCCJIT_H
#define MLIR_GCCJIT_TRANSLATION_TRANSLATETOGCCJIT_H
#include "mlir/IR/BuiltinOps.h"
#include <libgccjit.h>
namespace mlir::gccjit {

void registerToGCCJITGimpleTranslation();
void registerToGCCJITReproducerTranslation();

struct GCCJITContextReleaser {
  void operator()(gcc_jit_context *ctxt) const { gcc_jit_context_release(ctxt); }
};
using GCCJITContext = std::unique_ptr<gcc_jit_context, GCCJITContextReleaser>;

namespace impl {
struct GCCJITTypeConverter;
struct Translator;
} // namespace impl

class Translator;
class GCCJITTypeConverter {
  std::unique_ptr<impl::GCCJITTypeConverter> impl;

private:
  GCCJITTypeConverter(std::unique_ptr<impl::GCCJITTypeConverter>);

public:
  void convertTypes(mlir::TypeRange types, llvm::SmallVectorImpl<gcc_jit_type *> &result);
  gcc_jit_type *convertType(mlir::Type type);
  Translator &getTranslator();
  friend impl::Translator;
};

class Translator {
  std::unique_ptr<impl::Translator> impl;

public:
  gcc_jit_context *getContext() const;
  Translator();
  GCCJITContext takeContext();
  GCCJITTypeConverter &getTypeConverter();
  void translateModuleToGCCJIT(ModuleOp op);
};

llvm::Expected<GCCJITContext> translateModuleToGCCJIT(ModuleOp op);
} // namespace mlir::gccjit

#endif // MLIR_GCCJIT_TRANSLATION_TRANSLATETOGCCJIT_H
