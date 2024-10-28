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

#include "mlir-gccjit/IR/GCCJITTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/DenseMap.h"

#include "libgccjit.h"

namespace mlir::gccjit {

void registerToGCCJITGimpleTranslation();
void registerToGCCJITReproducerTranslation();
void registerToGCCJITAssemblyTranslation();
void registerToGCCJITObjectTranslation();
void registerToGCCJITExecutableTranslation();
void registerToGCCJITDylibTranslation();

struct GCCJITContextDeleter {
  void operator()(gcc_jit_context *ctxt) const;
};
using GCCJITContext = std::unique_ptr<gcc_jit_context, GCCJITContextDeleter>;

class GCCJITTranslation {
  class FunctionEntry;
  class StructEntry;
  class UnionEntry;

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

  gcc_jit_lvalue *getGlobalLValue(SymbolRefAttr symbol);
  FunctionEntry getFunction(SymbolRefAttr symbol);

private:
  class FunctionEntry {
    gcc_jit_function *fnHandle;

  public:
    FunctionEntry() : fnHandle(nullptr) {}
    FunctionEntry(gcc_jit_function *fnHandle) : fnHandle(fnHandle) {}
    operator gcc_jit_function *() const { return fnHandle; }
    size_t getParamCount() const {
      assert(fnHandle);
      return gcc_jit_function_get_param_count(fnHandle);
    }
    gcc_jit_param *operator[](size_t index) const {
      assert(fnHandle);
      return gcc_jit_function_get_param(fnHandle, index);
    }
  };

  class StructEntry {
    gcc_jit_struct *structHandle;

  public:
    StructEntry(gcc_jit_struct *structHandle) : structHandle(structHandle) {}
    operator gcc_jit_struct *() const { return structHandle; }
    size_t getFieldCount() const {
      return gcc_jit_struct_get_field_count(structHandle);
    }
    gcc_jit_field *operator[](size_t index) const {
      return gcc_jit_struct_get_field(structHandle, index);
    }
  };

  class UnionEntry {
    gcc_jit_type *unionHandle;
    llvm::SmallVector<gcc_jit_field *> fields;

  public:
    UnionEntry(gcc_jit_type *unionHandle, ArrayRef<gcc_jit_field *> fields)
        : unionHandle(unionHandle), fields(fields) {}
    operator gcc_jit_type *() const { return unionHandle; }
    size_t getFieldCount() const { return fields.size(); }
    gcc_jit_field *operator[](size_t index) const { return fields[index]; }
  };

  gcc_jit_context *ctxt;
  ModuleOp moduleOp;
  llvm::DenseMap<SymbolRefAttr, FunctionEntry> functionMap;
  llvm::DenseMap<SymbolRefAttr, gcc_jit_lvalue *> globalMap;
  llvm::DenseMap<Type, gcc_jit_type *> typeMap;
  llvm::DenseMap<StructType, StructEntry> structMap;
  llvm::DenseMap<UnionType, UnionEntry> unionMap;

  void populateGCCJITModuleOptions();
  void declareAllFunctionAndGlobals();
  void translateGlobalInitializers();
  void translateFunctions();
};

llvm::Expected<GCCJITContext> translateModuleToGCCJIT(ModuleOp op);

} // namespace mlir::gccjit

#endif // MLIR_GCCJIT_TRANSLATION_TRANSLATETOGCCJIT_H
