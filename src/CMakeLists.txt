add_mlir_dialect_library(MLIRGCCJIT
    GCCJITDialect.cpp
    GCCJITOps.cpp
    GCCJITTypes.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/mlir-gccjit

    DEPENDS
    MLIRGCCJITOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)