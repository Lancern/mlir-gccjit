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

#ifndef MLIR_GCCJIT_PASSES_H
#define MLIR_GCCJIT_PASSES_H

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include "mlir-gccjit/IR/GCCJITDialect.h"

namespace mlir::gccjit {

std::unique_ptr<Pass> createConvertFuncToGCCJITPass();
std::unique_ptr<Pass> createConvertArithToGCCJITPass();
std::unique_ptr<Pass> createConvertMemrefToGCCJITPass();

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "mlir-gccjit/Passes.h.inc"

} // namespace mlir::gccjit

#endif // MLIR_GCCJIT_PASSES_H
