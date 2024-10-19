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
