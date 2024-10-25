add_mlir_dialect_library(MLIRGCCJITInterfaces
    GCCJITRecordTypeInterface.cpp
  DEPENDS
    MLIRGCCJIT
    MLIRGCCJITRecordTypeInterfaceIncGen
  LINK_COMPONENTS
    Core
  LINK_LIBS PUBLIC
    MLIRGCCJIT
)
