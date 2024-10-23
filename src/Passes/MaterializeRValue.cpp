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

#include "mlir-gccjit/Passes.h"
#

using namespace mlir;
using namespace mlir::gccjit;

namespace {
struct GCCJITMaterializeRValue
    : public GCCJITMaterializeRValueBase<GCCJITMaterializeRValue> {
  using GCCJITMaterializeRValueBase::GCCJITMaterializeRValueBase;
  void runOnOperation() override final {}
};
} // namespace

std::unique_ptr<Pass> mlir::gccjit::createGCCJITMaterializeRValuePass() {
  return std::make_unique<GCCJITMaterializeRValue>();
}
