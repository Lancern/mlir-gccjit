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

#ifndef MLIR_GCCJIT_PASSES
#define MLIR_GCCJIT_PASSES

include "mlir/Pass/PassBase.td"

def ConvertFuncToGCCJIT : Pass<"convert-func-to-gccjit", "::mlir::ModuleOp"> {
  let summary = "Convert Functions and control flows to GCCJIT Dialect";
  let description = [{
    This pass converts function operations and control flow operations to GCCJIT
    dialect.
  }];
  let constructor = "::mlir::gccjit::createConvertFuncToGCCJITPass()";
  let dependentDialects = [
    "::mlir::gccjit::GCCJITDialect",
    "::mlir::func::FuncDialect",
    "::mlir::BuiltinDialect",
    "::mlir::cf::ControlFlowDialect"
  ];
}

def ConvertArithToGCCJIT : Pass<"convert-arith-to-gccjit", "::mlir::ModuleOp"> {
  let summary = "Convert Arithmetic Operations to GCCJIT Dialect";
  let description = [{
    `-convert-arith-to-gccjit` converts arithmetic operations to GCCJIT dialect.
  }];
  let constructor = "::mlir::gccjit::createConvertArithToGCCJITPass()";
  let dependentDialects = [
    "::mlir::gccjit::GCCJITDialect",
    "::mlir::arith::ArithDialect"
  ];
}

def ConvertMemrefToGCCJIT : Pass<"convert-memref-to-gccjit",
                                 "::mlir::ModuleOp"> {
  let summary = "Convert Memref Operations to GCCJIT Dialect";
  let description = [{
    `-convert-memref-to-gccjit` converts memref operations to GCCJIT dialect.
  }];
  let constructor = "::mlir::gccjit::createConvertMemrefToGCCJITPass()";
  let dependentDialects = [
    "::mlir::gccjit::GCCJITDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

#endif // MLIR_GCCJIT_PASSES
