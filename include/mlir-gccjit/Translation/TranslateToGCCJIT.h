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
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/DenseMap.h"

#include "libgccjit.h"

namespace mlir::gccjit {

void registerToGCCJITGimpleTranslation();
void registerToGCCJITReproducerTranslation();

struct GCCJITContextDeleter {
  void operator()(gcc_jit_context *ctxt) const;
};
using GCCJITContext = std::unique_ptr<gcc_jit_context, GCCJITContextDeleter>;

class GCCJITTranslation {
public:
  GCCJITTranslation();
  ~GCCJITTranslation();

  gcc_jit_context *getContext() const;
  GCCJITContext takeContext();

  MLIRContext *getMLIRContext() const { return moduleOp->getContext(); }

  void translateModuleToGCCJIT(ModuleOp op);

  gcc_jit_type *convertType(Type type);
  void convertTypes(TypeRange types, llvm::SmallVector<gcc_jit_type *> &result);

  gcc_jit_location *getLocation(LocationAttr loc);

private:
  struct FunctionEntry {
    gcc_jit_function *fnHandle;
    llvm::SmallVector<gcc_jit_param *> params;
  };

  struct StructEntry {
    gcc_jit_struct *structHandle;
    llvm::SmallVector<gcc_jit_field *> fields;
  };

  struct UnionEntry {
    gcc_jit_type *unionHandle;
    llvm::SmallVector<gcc_jit_field *> fields;
  };

  gcc_jit_context *ctxt;
  ModuleOp moduleOp;
  llvm::DenseMap<SymbolRefAttr, FunctionEntry> functionMap;
  llvm::DenseMap<SymbolRefAttr, gcc_jit_lvalue *> globalMap;
  llvm::DenseMap<Type, gcc_jit_type *> typeMap;
  llvm::DenseMap<Type, StructEntry> structMap;
  llvm::DenseMap<Type, UnionEntry> unionMap;

  void populateGCCJITModuleOptions();
  void declareAllFunctionAndGlobals();
};

llvm::Expected<GCCJITContext> translateModuleToGCCJIT(ModuleOp op);

} // namespace mlir::gccjit

#endif // MLIR_GCCJIT_TRANSLATION_TRANSLATETOGCCJIT_H
