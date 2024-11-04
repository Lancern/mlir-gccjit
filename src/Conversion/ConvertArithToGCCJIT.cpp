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

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include "mlir-gccjit/Conversion/Conversions.h"
#include "mlir-gccjit/Conversion/TypeConverter.h"
#include "mlir-gccjit/IR/GCCJITAttrs.h"
#include "mlir-gccjit/IR/GCCJITOps.h"
#include "mlir-gccjit/IR/GCCJITOpsEnums.h"
#include "mlir-gccjit/IR/GCCJITTypes.h"
#include "mlir-gccjit/Passes.h"

using namespace mlir;
using namespace mlir::gccjit;

namespace {
struct ConvertArithToGCCJITPass
    : public ConvertArithToGCCJITBase<ConvertArithToGCCJITPass> {
  using ConvertArithToGCCJITBase::ConvertArithToGCCJITBase;
  void runOnOperation() override final;
};

void ConvertArithToGCCJITPass::runOnOperation() {
  auto moduleOp = getOperation();
  auto typeConverter = GCCJITTypeConverter();
  mlir::RewritePatternSet patterns(&getContext());
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<gccjit::GCCJITDialect>();
  target.addIllegalDialect<mlir::arith::ArithDialect>();
  llvm::SmallVector<Operation *> ops;
  for (auto func : moduleOp.getOps<func::FuncOp>())
    ops.push_back(func);
  if (failed(applyPartialConversion(ops, target, std::move(patterns))))
    signalPassFailure();
}
} // namespace

std::unique_ptr<Pass> mlir::gccjit::createConvertArithToGCCJITPass() {
  return std::make_unique<ConvertArithToGCCJITPass>();
}
