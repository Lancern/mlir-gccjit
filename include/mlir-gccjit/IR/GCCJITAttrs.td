#ifndef MLIR_GCCJIT_IR_GCCJIT_ATTRS
#define MLIR_GCCJIT_IR_GCCJIT_ATTRS

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir-gccjit/IR/GCCJITDialect.td"

//===----------------------------------------------------------------------===//
// GCCJIT Attrs
//===----------------------------------------------------------------------===//

class GCCJIT_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<GCCJIT_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class GCCJIT_UnitAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : GCCJIT_Attr<name, attrMnemonic, traits> {
  let returnType = "bool";
  let defaultValue = "false";
  let valueType = NoneType;
  let isOptional = 1;
}

//===----------------------------------------------------------------------===//
// TLS Model Attrs
//===----------------------------------------------------------------------===//

def TLSModel_None : I32EnumAttrCase<"None", 0, "none">;
def TLSModel_GlobalDynamic : I32EnumAttrCase<"GlobalDynamic", 1, "global_dynamic">;
def TLSModel_LocalDynamic : I32EnumAttrCase<"LocalDynamic", 2, "local_dynamic">;
def TLSModel_InitialExec : I32EnumAttrCase<"InitialExec", 3, "initial_exec">;
def TLSModel_LocalExec : I32EnumAttrCase<"LocalExec", 4, "local_exec">;

def TLSModelEnum : I32EnumAttr<"TLSModelEnum", "TLS model",
                          [TLSModel_None, TLSModel_GlobalDynamic,
                           TLSModel_LocalDynamic, TLSModel_InitialExec,
                           TLSModel_LocalExec]> {
  let cppNamespace = "mlir::gccjit";
}

def TLSModelAttr : GCCJIT_Attr<"TLSModel", "tls_model"> {
  let summary = "Thread Local Storage model";
  let parameters = (ins "TLSModelEnumAttr":$model);
  let description = [{
    Represents the Thread Local Storage model for the lvalue.
  }];
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    bool isNone() const { return getModel().getValue() == TLSModelEnum::None; }
    bool isGlobalDynamic() const { return getModel().getValue() == TLSModelEnum::GlobalDynamic; }
    bool isLocalDynamic() const { return getModel().getValue() == TLSModelEnum::LocalDynamic; }
    bool isInitialExec() const { return getModel().getValue() == TLSModelEnum::InitialExec; }
    bool isLocalExec() const { return getModel().getValue() == TLSModelEnum::LocalExec; }
  }];
  let genVerifyDecl = 0;
}

//===----------------------------------------------------------------------===//
// Integer Attrs
//===----------------------------------------------------------------------===//
def IntAttr : GCCJIT_Attr<"Int", "int", [TypedAttrInterface]> {
  let summary = "An attribute containing a integer value";
  let description = [{
    An `#gccjit.int` attribute represents a constant integer value of some
    libgccjit integer type.

    Example:
    ```mlir
    #gccjit.int<42>
    ```
  }];
  let parameters = (ins
    AttributeSelfTypeParameter<"", "mlir::gccjit::IntType">:$type,
    "::mlir::APInt":$value
  );
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Floating-point Attrs
//===----------------------------------------------------------------------===//
def FloatAttr : GCCJIT_Attr<"Float", "float", [TypedAttrInterface]> {
  let summary = "An attribute containing a constant floating-point value";
  let description = [{
    A `#gccjit.float` attribute represents a constant floating-point value of
    some libgccjit floating-point type.

    Example:
    ```mlir
    #gccjit.float<42.0>
    ```
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "mlir::gccjit::FloatType">:$type,
    "::mlir::APFloat":$value
  );
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Special constant attributes
//===----------------------------------------------------------------------===//

def ZeroAttr : GCCJIT_Attr<"Zero", "zero", [TypedAttrInterface]> {
  let summary = "An attribute containing a constant zero";
  let description = [{
    A `#gccjit.zero` attribute represents a constant zero value of some
    libgccjit integer or floating-point type.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);

  let assemblyFormat = "";
  let genVerifyDecl = 1;
}

def OneAttr : GCCJIT_Attr<"One", "one", [TypedAttrInterface]> {
  let summary = "An attribute containing a constant one";
  let description = [{
    A `#gccjit.one` attribute represents a constant one value of some
    libgccjit integer or floating-point type.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);

  let assemblyFormat = "";
  let genVerifyDecl = 1;
}

def NullAttr : GCCJIT_Attr<"Null", "null", [TypedAttrInterface]> {
  let summary = "An attribute containing a null pointer constant";
  let description = [{
    A `#gccjit.null` attribute represents a constant null pointer value of some
    libgccjit pointer type.
  }];

  let parameters = (ins
      AttributeSelfTypeParameter<"", "mlir::gccjit::PointerType">:$type);

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// SourceLoc Attr
//===----------------------------------------------------------------------===//

def SourceLocAttr : GCCJIT_Attr<"SourceLoc", "loc"> {
  let summary = "Source location";
  let description = [{
    Represent a source location in GCCJIT.
  }];
  let parameters = (ins "mlir::StringAttr":$filename, "int":$line,
                        "int":$column);
  let assemblyFormat = [{
    `<` $filename `:` $line `:` $column `>`
  }];
}

//===----------------------------------------------------------------------===//
// Record field attribute
//===----------------------------------------------------------------------===//

def FieldAttr : GCCJIT_Attr<"Field", "field"> {
  let summary = "A field within a struct or a union";
  let description = [{
    The `#gccjit.field` attribute represents a field within a record (i.e. a
    record or a union).

    When the `bitWidth` parameter is present, the attribute represents a
    bitfield. The `bitWidth` parameter gives the width of the bitfield.
  }];

  let parameters = (ins "mlir::StringAttr":$name, "mlir::Type":$type,
                        OptionalParameter<"unsigned">:$bitWidth,
                        OptionalParameter<"mlir::gccjit::SourceLocAttr">:$loc);

  let builders = [
    AttrBuilder<(ins "mlir::StringAttr":$name, "mlir::Type":$type), [{
      return get($_ctxt, name, type, 0, mlir::gccjit::SourceLocAttr{});
    }]>,
    AttrBuilder<(ins "mlir::StringAttr":$name, "mlir::Type":$type,
                     "unsigned":$bitWidth), [{
      return get($_ctxt, name, type, bitWidth, mlir::gccjit::SourceLocAttr{});
    }]>,
    AttrBuilder<(ins "mlir::StringAttr":$name, "mlir::Type":$type,
                     "mlir::gccjit::SourceLocAttr":$loc), [{
      return get($_ctxt, name, type, 0, loc);
    }]>,
  ];

  let assemblyFormat = [{
    `<` $type $name (`:` $bitWidth^)? `>` ($loc^)?
  }];
}

//===----------------------------------------------------------------------===//
// OptLevel Attr
//===----------------------------------------------------------------------===//

def OptLevel_0 : I32EnumAttrCase<"O0", 0, "O0">;
def OptLevel_1 : I32EnumAttrCase<"O1", 1, "O1">;
def OptLevel_2 : I32EnumAttrCase<"O2", 2, "O2">;
def OptLevel_3 : I32EnumAttrCase<"O3", 3, "O3">;
def OptLevelEnum : I32EnumAttr<"OptLevelEnum", "Optimization level",
                          [OptLevel_0, OptLevel_1, OptLevel_2, OptLevel_3]> {
  let cppNamespace = "mlir::gccjit";
}
def OptLevelAttr : GCCJIT_Attr<"OptLevel", "opt_level"> {
  let summary = "Optimization level";
  let parameters = (ins "OptLevelEnumAttr":$level);
  let description = [{
    Represents the optimization level.
  }];
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    bool isO0() const { return getLevel().getValue() == OptLevelEnum::O0; }
    bool isO1() const { return getLevel().getValue() == OptLevelEnum::O1; }
    bool isO2() const { return getLevel().getValue() == OptLevelEnum::O2; }
    bool isO3() const { return getLevel().getValue() == OptLevelEnum::O3; }
  }];
  let genVerifyDecl = 0;
}

//===----------------------------------------------------------------------===//
// Function Attributes
//===----------------------------------------------------------------------===//
def FnAttr_Alias : I32EnumAttrCase<"Alias", 0, "alias">;
def FnAttr_AlwaysInline : I32EnumAttrCase<"AlwaysInline", 1, "always_inline">;
def FnAttr_Inline : I32EnumAttrCase<"Inline", 2, "inline">;
def FnAttr_NoInline : I32EnumAttrCase<"NoInline", 3, "noinline">;
def FnAttr_Target : I32EnumAttrCase<"Target", 4, "target">;
def FnAttr_Used : I32EnumAttrCase<"Used", 5, "used">;
def FnAttr_Visibility : I32EnumAttrCase<"Visibility", 6, "visibility">;
def FnAttr_Cold : I32EnumAttrCase<"Cold", 7, "cold">;
def FnAttr_ReturnsTwice : I32EnumAttrCase<"ReturnsTwice", 8, "returns_twice">;
def FnAttr_Pure : I32EnumAttrCase<"Pure", 9, "pure">;
def FnAttr_Const : I32EnumAttrCase<"Const", 10, "const">;
def FnAttr_Weak : I32EnumAttrCase<"Weak", 11, "weak">;
def FnAttr_Nonnull : I32EnumAttrCase<"Nonnull", 12, "nonnull">;
def FnAttrEnum : I32EnumAttr<"FnAttrEnum", "Function attribute",
                          [FnAttr_Alias, FnAttr_AlwaysInline, FnAttr_Inline,
                           FnAttr_NoInline, FnAttr_Target, FnAttr_Used,
                           FnAttr_Visibility, FnAttr_Cold, FnAttr_ReturnsTwice,
                           FnAttr_Pure, FnAttr_Const, FnAttr_Weak, FnAttr_Nonnull]> {
  let cppNamespace = "mlir::gccjit";
}

def FnAttr : GCCJIT_Attr<"Function", "fn_attr"> {
  let summary = "Function Attribute";
  let description = [{
    A `#gccjit.fn_attr` attribute represents a function attribute.
    There three types of function attributes:
    - unit attributes: those without a value.
    - string attributes: those with a string value.
    - integer array attributes: those with an array of integers.
  }];

  let parameters = (ins
    "FnAttrEnumAttr":$attr,
    OptionalParameter<"::mlir::StringAttr">:$strValue,
    OptionalParameter<"::mlir::DenseI32ArrayAttr">:$intArrayValue
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 0;
}

//===----------------------------------------------------------------------===//
// ByteArrayInitializer Attr
//===----------------------------------------------------------------------===//
def ByteArrayInitializerAttr : GCCJIT_Attr<"ByteArrayInitializer", "byte_array"> {
  let summary = "Byte array initializer";
  let parameters = (ins "DenseI8ArrayAttr":$initializer);
  let description = [{
    Represents the byte array initializer.
  }];
  let assemblyFormat = [{
    `<` $initializer `>`
  }];
}
#endif // MLIR_GCCJIT_IR_GCCJIT_ATTRS


