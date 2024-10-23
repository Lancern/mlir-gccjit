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

def GCCJITMaterializeRValue : Pass<"gccjit-materialize-rvalue", "::mlir::gccjit::FuncOp"> {
  let summary = "Automatically materialize rvalue expressions in GCCJIT dialect";
  let description = [{
    RValue expressions are lazy in GCCJIT. If not associated with a local variable, 
    they will never be emitted inplace. Using a nonmaterialized rvalue expression twice will
    result in two separate computations. Using a stateful rvalue across an effectful operation
    may result in different results. GCCJIT dialect mimic the low-level behavior of GCCJIT on default,
    translating rvalue/lvalue expression plainly. However, if one wants to use this dialect similar to
    the traditional SSA semantics, this pass will insert nessary materialization for rvalue expressions.

    Definition (Constant RValue): An RValue expression is constant if and only if it is created by
    a constant operation.

    The materialization criteria are as follows:
    - If the RValue expression is a mere constant, then it will not be materialized.
    - Otherwise:
        - if the RValue has a user that is not local to the block, then the expression is materialized.
        - Otherwise, inside a block:
          - If the RValue has multiple uses, it shall be materialized.
          - If the RValue expression is immediately consumed by terminators, then there is no need to materialize.
          - Otherwise:
            - If the RValue expression is a call to non-pure/const annotated function, then the expression is immediately 
                materialized.
            - If the RValue expression is a load/dereferece from a volatile lvalue or a dereference of a volatile pointer, then the expression
                is immediately materialized.
            - If the RValue expression's def-use chain overlap with any other effectful operation such as function call, asm block, or assignment 
                to variable that is not created by the materialization process, then the expression is materialized.

    By materialization, we mean that a local variable will be created and singly assigned by the RValue expression. All
    future uses of the RValue expression will be replaced by the local variable.
  }];
  let constructor = "::mlir::gccjit::createGCCJITMaterializeRValuePass()";
  let dependentDialects = ["::mlir::gccjit::GCCJITDialect"];
}

#endif // MLIR_GCCJIT_PASSES
