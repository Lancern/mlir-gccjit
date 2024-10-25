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

#include "mlir-gccjit/IR/GCCJITAttrs.h"
#include "mlir-gccjit/IR/GCCJITOps.h"
#include "mlir-gccjit/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::gccjit;

namespace {

bool isExpr(Operation *op) {
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<ConstantOp, LiteralOp, SizeOfOp, AlignOfOp, AsRValueOp, BinaryOp,
            UnaryOp, CompareOp, CallOp, CastOp, BitCastOp, PtrCallOp, AddrOp,
            FnAddrOp, GetGlobalOp>([&](auto) { return true; })
      .Default([&](Operation *) { return false; });
}

class GCCJITMaterializeRValue
    : public GCCJITMaterializeRValueBase<GCCJITMaterializeRValue> {
private:
  void fullMaterialize(FuncOp func);
  void materializeOp(IRRewriter &rewriter, AsmState &state, Operation *op);
  void materializeOps(IRRewriter &rewriter, AsmState &state,
                      const llvm::SmallVectorImpl<Operation *> &ops);

public:
  using GCCJITMaterializeRValueBase::GCCJITMaterializeRValueBase;

  void runOnOperation() override final {
    if (!full)
      llvm_unreachable("automatic materialization not implemented");
    fullMaterialize(getOperation());
  }

  void enableFullMaterialization() { full = true; }
};
} // namespace

void GCCJITMaterializeRValue::materializeOp(IRRewriter &rewriter,
                                            AsmState &state, Operation *op) {
  if (op->hasAttr("gccjit.eval"))
    return;
  if (op->getNumResults() == 1) {
    rewriter.setInsertionPointAfter(op);
    auto value = op->getResult(0);
    auto lvalueTy = LValueType::get(op->getContext(), value.getType());
    std::string name;
    std::string buffer;
    llvm::raw_string_ostream bufferStream(buffer);
    value.printAsOperand(bufferStream, state);
    bufferStream.flush();
    name = "var";
    for (auto &c : buffer)
      if (isalnum(c))
        name.push_back(c);
    auto var = rewriter.create<LocalOp>(
        op->getLoc(), lvalueTy, nullptr, nullptr, nullptr,
        StringLiteralAttr::get(op->getContext(), rewriter.getStringAttr(name)));
    auto assign = rewriter.create<AssignOp>(op->getLoc(), value, var);
    auto loaded =
        rewriter.create<AsRValueOp>(op->getLoc(), value.getType(), var);
    rewriter.replaceAllUsesExcept(value, loaded, assign);
  } else {
    op->setAttr("gccjit.eval", rewriter.getUnitAttr());
  }
};

void GCCJITMaterializeRValue::materializeOps(
    IRRewriter &rewriter, AsmState &state,
    const llvm::SmallVectorImpl<Operation *> &ops) {
  for (auto *op : ops)
    materializeOp(rewriter, state, op);
}

void GCCJITMaterializeRValue::fullMaterialize(FuncOp func) {
  IRRewriter rewriter(func->getContext());
  llvm::SmallVector<Operation *> ops;
  AsmState state(func);
  func.walk([&](Operation *op) {
    if (isExpr(op))
      ops.push_back(op);
  });
  materializeOps(rewriter, state, ops);
}

std::unique_ptr<Pass> mlir::gccjit::createGCCJITMaterializeRValuePass() {
  return std::make_unique<GCCJITMaterializeRValue>();
}
