#ifndef GCCJIT_TYPES
#define GCCJIT_TYPES

include "mlir/IR/AttrTypeBase.td"

include "mlir-gccjit/IR/GCCJITDialect.td"

class GCCJIT_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<GCCJIT_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

#endif // GCCJIT_TYPES
