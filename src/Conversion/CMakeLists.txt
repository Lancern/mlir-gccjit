add_mlir_conversion_library(MLIRGCCJITConversion
    TypeConverter.cpp
    ConvertFuncToGCCJIT.cpp
    ConvertArithToGCCJIT.cpp
    ConvertMemrefToGCCJIT.cpp

    DEPENDS
    MLIRGCCJIT
    MLIRGCCJITPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRGCCJIT
    libgccjit

    LINK_COMPONENTS
    Core
)
