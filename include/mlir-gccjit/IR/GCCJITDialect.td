#ifndef GCCJIT_DIALECT
#define GCCJIT_DIALECT

include "mlir/IR/OpBase.td"

def GCCJIT_Dialect : Dialect {
  let name = "gccjit";

  let summary = "Dialect for libgccjit";
  let description = [{
    The gccjit dialect allows you to generate code with libgccjit through MLIR.
  }];

  let cppNamespace = "::mlir::gccjit";

  let dependentDialects = [
    "::mlir::BuiltinDialect",
    "::mlir::func::FuncDialect",
  ];

  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

class GCCJIT_Op<string mnemonic, list<Trait> traits = []>
    : Op<GCCJIT_Dialect, mnemonic, traits>;

#endif // GCCJIT_DIALECT
