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
def IntAttr : GCCJIT_Attr<"Int", "int"> {
  let summary = "An Attribute containing a integer value";
  let description = [{
    Integer literal attribute.
  }];
  let parameters = (ins "::mlir::APInt":$value, "bool":$isLong);
  let hasCustomAssemblyFormat = 1;
}
#endif // MLIR_GCCJIT_IR_GCCJIT_ATTRS


