add_mlir_dialect(GCCJITOps gccjit)

# Generate extra headers for custom enum and attrs.
set(LLVM_TARGET_DEFINITIONS GCCJITOps.td)
mlir_tablegen(GCCJITOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(GCCJITOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(GCCJITOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(GCCJITOpsAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRGCCJITEnumsGen)
add_dependencies(mlir-headers MLIRGCCJITEnumsGen)

add_mlir_gccjit_doc(GCCJITDialect GCCJITDialect gccjit/ -gen-dialect-doc)
add_mlir_gccjit_doc(GCCJITOps GCCJITOps gccjit/ -gen-op-doc)
add_mlir_gccjit_doc(GCCJITAttrs GCCJITAttrs gccjit/ -gen-attrdef-doc)
add_mlir_gccjit_doc(GCCJITTypes GCCJITTypes gccjit/ -gen-typedef-doc)

add_subdirectory(Interfaces)
