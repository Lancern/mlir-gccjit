#ifndef GCCJIT_TYPES
#define GCCJIT_TYPES

include "mlir/IR/AttrTypeBase.td"

include "mlir-gccjit/IR/GCCJITDialect.td"
include "mlir-gccjit/IR/Interfaces/GCCJITRecordTypeInterface.td"

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

//===----------------------------------------------------------------------===//
// FuncType
//===----------------------------------------------------------------------===//

def GCCJIT_FuncType : GCCJIT_Type<"Func", "func"> {
  let summary = "GCCJIT function type";
  let description = [{
    The `!gccjit.func` is a function type. It consists of a single return type, a
    list of parameter types and can optionally be variadic.
  }];

  let parameters = (ins ArrayRefParameter<"Type">:$inputs, "Type":$returnType,
                        "bool":$varArg);
  let assemblyFormat = [{
    `<` $returnType ` ` `(` custom<FuncTypeArgs>($inputs, $varArg) `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<Type>":$inputs, "Type":$returnType,
      CArg<"bool", "false">:$isVarArg), [{
      return $_get(returnType.getContext(), inputs, returnType, isVarArg);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Returns whether the function is variadic.
    bool isVarArg() const { return getVarArg(); }

    /// Returns the `i`th input operand type. Asserts if out of bounds.
    Type getInput(unsigned i) const { return getInputs()[i]; }

    /// Returns the number of arguments to the function.
    unsigned getNumInputs() const { return getInputs().size(); }

    /// Returns the result type of the function as an ArrayRef, enabling better
    /// integration with generic MLIR utilities.
    ArrayRef<Type> getReturnTypes() const;

    /// Returns whether the function is returns void.
    bool isVoid() const;

    /// Returns a clone of this function type with the given argument
    /// and result types.
    FuncType clone(TypeRange inputs, TypeRange results) const;
  }];
}

//===----------------------------------------------------------------------===//
// Void type
//===----------------------------------------------------------------------===//

def GCCJIT_VoidType : GCCJIT_Type<"Void", "void"> {
  let summary = "GCCJIT void type";
  let description = [{
    The `!gccjit.void` type represents the C/C++ `void` type.
  }];
  let extraClassDeclaration = [{
    /// Returns a clone of this type with the given context.
    std::string getAlias() const { return "void"; };
  }];
}

//===----------------------------------------------------------------------===//
// Pointer type
//===----------------------------------------------------------------------===//

def GCCJIT_PointerType : GCCJIT_Type<"Pointer", "ptr"> {
  let summary = "GCCJIT pointer type";
  let description = [{
    The `!gccjit.ptr` type represents a pointer type. It consists of a single
    element type.
  }];
  let parameters = (ins "mlir::Type":$elementType);
  let assemblyFormat = [{
    `<` $elementType `>`
  }];
}

//===----------------------------------------------------------------------===//
// Qualified type
//===----------------------------------------------------------------------===//

def GCCJIT_QualifiedType : GCCJIT_Type<"Qualified", "qualified"> {
  let summary = "GCCJIT qualified type";
  let description = [{
    GCCJIT type can be qualified with `const`, `restrict`, `volatile`.
  }];
  let parameters = (ins "mlir::Type":$elementType, "bool":$isConst,
                        "bool":$isRestrict, "bool":$isVolatile);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Integer type
//===----------------------------------------------------------------------===//

def GCCJIT_IntType : GCCJIT_Type<"Int", "int"> {
  let summary = "Integer type";
  let description = [{
    GCCJIT type that represents integer types with arbitrary precision:
    - GCCJIT type can be either explicitly sized or not.
    - GCCJIT type can be either signed or unsigned.
  }];

  let parameters = (ins "::gcc_jit_types":$kind);
  let hasCustomAssemblyFormat = 1;
}

class SpecificInt<string enumValue>
  : Type<And<[
        CPred<"::mlir::isa<::mlir::gccjit::IntType>($_self)">,
        CPred<"::mlir::cast<::mlir::gccjit::IntType>($_self).getKind() == " # enumValue>
        ]>, "integer with " #enumValue, "::mlir::gccjit::IntType">,
    BuildableType<
      "mlir::gccjit::IntType::get($_builder.getContext(), "
      # enumValue # ")"> {
}

def GCCJIT_BoolType : SpecificInt<"GCC_JIT_TYPE_BOOL">;

//===----------------------------------------------------------------------===//
// Float type
//===----------------------------------------------------------------------===//

def GCCJIT_FloatType : GCCJIT_Type<"Float", "fp"> {
  let summary = "Floating point type";
  let description = [{
    GCCJIT type that represents floating point types
  }];

  let parameters = (ins "::gcc_jit_types":$kind);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Complex type
//===----------------------------------------------------------------------===//

def GCCJIT_ComplexType : GCCJIT_Type<"Complex", "complex"> {
  let summary = "Complex type";
  let description = [{
    GCCJIT type that represents complex types
  }];

  let parameters = (ins "::gcc_jit_types":$kind);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Array type
//===----------------------------------------------------------------------===//
def GCCJIT_ArrayType : GCCJIT_Type<"Array", "array"> {
  let summary = "GCCJIT array type";
  let description = [{
    The `!gccjit.array` type represents an array type. It consists of a single
    element type and a constant length.
  }];
  let parameters = (ins "mlir::Type":$elementType, "unsigned":$size);
  let assemblyFormat = [{
    `<` $elementType `,` $size `>`
  }];
}

//===----------------------------------------------------------------------===//
// Vector type
//===----------------------------------------------------------------------===//
def GCCJIT_VectorType : GCCJIT_Type<"Vector", "vector"> {
  let summary = "GCCJIT vector type";
  let description = [{
    The `!gccjit.vector` type represents a vector type. It consists of a single
    element type and a constant length.
  }];
  let parameters = (ins "mlir::Type":$elementType, "unsigned":$numUnits);
  let assemblyFormat = [{
    `<` $elementType `,` $numUnits `>`
  }];
}

//===----------------------------------------------------------------------===//
// Struct and union type
//===----------------------------------------------------------------------===//
class GCCJIT_RecordType<string name, string typeMnemonic>
    : GCCJIT_Type<name, typeMnemonic, [
        DeclareTypeInterfaceMethods<GCCJITRecordTypeInterface>
      ]> {
  let parameters = (ins "mlir::StringAttr":$name,
                        "mlir::ArrayAttr":$fields);
  let assemblyFormat = [{
    `<` custom<RecordBody>($name, $fields) `>`
  }];

  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$name,
                     "llvm::ArrayRef<mlir::gccjit::FieldAttr>":$fields), [{
      auto nameAttr = StringAttr::get($_ctxt, name);
      auto fieldsAttr = ArrayAttr::get(
          $_ctxt, llvm::SmallVector<Attribute>(fields.begin(), fields.end()));
      return get($_ctxt, nameAttr, fieldsAttr);
    }]>
  ];

  let genVerifyDecl = 1;
}

def GCCJIT_StructType : GCCJIT_RecordType<"Struct", "struct"> {
  let summary = "GCCJIT struct type";
  let description = [{
    The `!gccjit.struct` type represents a GCCJIT struct type.
  }];
}

def GCCJIT_UnionType : GCCJIT_RecordType<"Union", "union"> {
  let summary = "GCCJIT union type";
  let description = [{
    The `!gccjit.union` type represents a GCCJIT union type.
  }];
}

#endif // GCCJIT_TYPES
