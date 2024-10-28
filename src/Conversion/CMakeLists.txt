get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
add_mlir_dialect_library(MLIRGCCJITConversion
    TypeConverter.cpp

    DEPENDS
    MLIRGCCJIT
    ${dialect_libs}
    ${conversion_libs}

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRGCCJIT
    libgccjit
    ${dialect_libs}
    ${conversion_libs}

    LINK_COMPONENTS
    Core
)
