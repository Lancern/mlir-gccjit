function(add_mlir_gccjit_doc doc_filename output_file output_directory command)
    set(LLVM_TARGET_DEFINITIONS ${doc_filename}.td)
    tablegen(MLIR ${output_file}.md ${command} ${ARGN})
    set(GEN_DOC_FILE ${PROJECT_BINARY_DIR}/docs/${output_directory}${output_file}.md)
    add_custom_command(
            OUTPUT ${GEN_DOC_FILE}
            COMMAND ${CMAKE_COMMAND} -E copy
                    ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md
                    ${GEN_DOC_FILE}
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md)
    add_custom_target(${output_file}DocGen DEPENDS ${GEN_DOC_FILE})
    add_dependencies(mlir-doc ${output_file}DocGen)
endfunction()

add_subdirectory(IR)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name GCCJIT)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix GCCJIT)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix GCCJIT)
add_public_tablegen_target(MLIRGCCJITPassIncGen)

add_mlir_gccjit_doc(Passes GCCJITPasses gccjit/ -gen-pass-doc)

add_custom_target(link-mlir-docs
    COMMAND ${PROJECT_SOURCE_DIR}/www/link-docs.sh ${PROJECT_BINARY_DIR}
    DEPENDS mlir-doc
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/www
)
