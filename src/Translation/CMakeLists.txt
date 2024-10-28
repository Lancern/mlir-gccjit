add_mlir_dialect_library(MLIRGTranslateToGCCJIT
    TranslateToGCCJIT.cpp
    Registration.cpp

    DEPENDS
    MLIRGCCJIT
    MLIRGCCJITInterfaces

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRGCCJIT
    MLIRGCCJITInterfaces
    libgccjit

    LINK_COMPONENTS
    Core
)
