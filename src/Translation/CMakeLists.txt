add_mlir_dialect_library(MLIRGTranslateToGCCJIT
    TranslateToGCCJIT.cpp
    Registration.cpp

    DEPENDS
    MLIRGCCJIT

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRGCCJIT
    libgccjit

    LINK_COMPONENTS
    Core
)
