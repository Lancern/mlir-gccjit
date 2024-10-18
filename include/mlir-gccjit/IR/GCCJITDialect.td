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

  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;

  let dependentDialects = [
    "::mlir::BuiltinDialect",
    "::mlir::func::FuncDialect",
  ];

  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
    Type parseType(DialectAsmParser &parser) const override;
    void printType(Type type, DialectAsmPrinter &printer) const override;
    Attribute parseAttribute(DialectAsmParser &parser,
                             Type type) const override;

    void printAttribute(Attribute attr, DialectAsmPrinter &os) const override;
  }];
}

class GCCJIT_Op<string mnemonic, list<Trait> traits = []>
    : Op<GCCJIT_Dialect, mnemonic, traits>;

#endif // GCCJIT_DIALECT
