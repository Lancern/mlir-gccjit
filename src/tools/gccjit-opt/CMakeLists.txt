get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_mlir_tool(reuse-opt
  main.cpp
  DEPENDS
    MLIROptLib
    MLIRGCCJIT
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
)
target_link_libraries(reuse-opt 
  PRIVATE
    MLIROptLib
    MLIRGCCJIT
    ${dialect_libs} 
    ${conversion_libs} 
    ${extension_libs} 
)
