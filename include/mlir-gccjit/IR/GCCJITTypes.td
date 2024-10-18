#ifndef GCCJIT_TYPES
#define GCCJIT_TYPES

include "mlir/IR/AttrTypeBase.td"

include "mlir-gccjit/IR/GCCJITDialect.td"

class GCCJIT_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<GCCJIT_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def GCCJIT_LValueType : GCCJIT_Type<"LValue", "lvalue"> {
    let cppNamespace = "mlir::gccjit";
    let summary = "LValue type";
    let description = [{
        `gccjit.lvalue` represents a lvalue in the meaning of C. It behaves
        like a reference to a mutable memory location.
    }];
}

#endif // GCCJIT_TYPES
