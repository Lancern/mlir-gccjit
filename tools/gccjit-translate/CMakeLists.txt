set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_mlir_tool(gccjit-translate
  main.cpp
  )
llvm_update_compile_flags(gccjit-translate)
target_link_libraries(gccjit-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRTranslateLib
  MLIRSupport
  MLIRTranslateToGCCJIT
  MLIRGCCJIT
  )

mlir_check_link_libraries(gccjit-translate)

add_dependencies(gccjit-tools gccjit-translate)
