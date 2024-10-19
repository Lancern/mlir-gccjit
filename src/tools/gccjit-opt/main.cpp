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

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir-gccjit/IR/GCCJITDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry Registry;
  mlir::registerAllDialects(Registry);
  Registry.insert<mlir::gccjit::GCCJITDialect>();
  mlir::registerAllExtensions(Registry);
  mlir::registerAllPasses();
  return failed(
      mlir::MlirOptMain(argc, argv, "GCCJIT analysis and optimization driver\n", Registry));
}
