#ifndef GCCJIT_TYPES
#define GCCJIT_TYPES

include "mlir/IR/AttrTypeBase.td"

include "mlir-gccjit/IR/GCCJITDialect.td"

class GCCJIT_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<GCCJIT_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def GCCJIT_LValueType : GCCJIT_Type<"LValue", "lvalue"> {
    let summary = "LValue type";
    let description = [{
        `gccjit.lvalue` represents a lvalue in the meaning of C. It behaves
        like a reference to a mutable memory location. Lvalues are addressable,
        so one can get a pointer from them. Lvalue's subfield can be projected as
        a new lvalue, while rvalue's subfield can be projected as a new rvalue.
        There are generally three ways to obtain an lvalue:
        1. create a local lvalue with respect to current function
        2. create a global lvalue with respect to current module
        3. function parameters in gimple are also lvalues

        See also [`gccjit::lvalue`](https://gcc.gnu.org/onlinedocs/jit/cp/topics/expressions.html#_CPPv4N6gccjit6lvalueE).
    }];
    let parameters = (ins "mlir::Type":$innerType);
    let assemblyFormat = [{
        `<` $innerType `>`
    }];
}

#endif // GCCJIT_TYPES
