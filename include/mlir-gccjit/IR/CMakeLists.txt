add_mlir_dialect(GCCJITOps gccjit)
add_mlir_doc(GCCJITDialect GCCJITDialect GCCJIT/ -gen-dialect-doc)
add_mlir_doc(GCCJITOps GCCJITOps GCCJIT/ -gen-op-doc)


# Generate extra headers for custom enum and attrs.
set(LLVM_TARGET_DEFINITIONS GCCJITOps.td)
mlir_tablegen(GCCJITOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(GCCJITOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(GCCJITOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(GCCJITOpsAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRGCCJITEnumsGen)
add_dependencies(mlir-headers MLIRGCCJITEnumsGen)
