add_mlir_dialect_library(MLIRGCCJITPasses
    MaterializeRValue.cpp
  DEPENDS
    MLIRGCCJIT
    MLIRGCCJITPassIncGen
  LINK_COMPONENTS
    Core
  LINK_LIBS PUBLIC
    MLIRGCCJIT
)
