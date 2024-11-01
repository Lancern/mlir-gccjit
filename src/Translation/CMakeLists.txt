add_mlir_dialect_library(MLIRTranslateToGCCJIT
    TranslateToGCCJIT.cpp
    Registration.cpp
    TypeTranslation.cpp

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
