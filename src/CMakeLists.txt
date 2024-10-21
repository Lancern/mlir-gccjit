add_mlir_dialect_library(MLIRGCCJIT
    GCCJITAttrs.cpp
    GCCJITDialect.cpp
    GCCJITOps.cpp
    GCCJITTypes.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/mlir-gccjit

    DEPENDS
    MLIRGCCJITOpsIncGen
    MLIRGCCJITEnumsGen

    LINK_LIBS PUBLIC
    MLIRIR
    libgccjit

    LINK_COMPONENTS
    Core
)

add_subdirectory(tools)
add_subdirectory(Translation)
