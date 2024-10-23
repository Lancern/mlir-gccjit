add_subdirectory(IR)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name GCCJIT)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix GCCJIT)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix GCCJIT)
add_public_tablegen_target(MLIRGCCJITPassIncGen)

add_mlir_doc(Passes GCCJITPasses ./ -gen-op-doc)
