#ifndef MLIR_GCCJIT_IR_INTERFACES_GCCJIT_RECORD_TYPE_INTERFACE
#define MLIR_GCCJIT_IR_INTERFACES_GCCJIT_RECORD_TYPE_INTERFACE

include "mlir/IR/OpBase.td"

def GCCJITRecordTypeInterface : TypeInterface<"GCCJITRecordTypeInterface"> {
  let description = [{
    Contains helper functions to query properties about a GCCJIT record type.

    A record type is either a struct type or a union type.
  }];
  let cppNamespace = "::mlir::gccjit";

  let methods = [
    InterfaceMethod<
      "Determine whether this record is a union",
      /*retTy=*/"bool",
      /*methodName=*/"isUnion"
    >,
    InterfaceMethod<
      "Get the name of this record",
      /*retTy=*/"llvm::StringRef",
      /*methodName=*/"getRecordName"
    >,
    InterfaceMethod<
      [{
        Get the fields in this record, as an ArrayAttr attribute whose elements
        are FieldAttr attributes.
      }],
      /*retTy=*/"ArrayAttr",
      /*methodName=*/"getRecordFields"
    >,
    InterfaceMethod<
      [{
        Get the source location of this record.
      }],
      /*retTy=*/"mlir::gccjit::SourceLocAttr",
      /*methodName=*/"getRecordLoc"
    >
  ];
}

#endif // MLIR_GCCJIT_IR_INTERFACES_GCCJIT_RECORD_TYPE_INTERFACE
