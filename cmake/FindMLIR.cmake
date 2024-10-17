include_guard(GLOBAL)

set(find_mlir_hints)
if (LLVM_DIR)
    list(APPEND find_mlir_hints "${LLVM_DIR}/..")
endif ()

find_package(MLIR
    REQUIRED CONFIG
    HINTS ${find_mlir_hints}
)
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION} at ${MLIR_DIR}")

include(${MLIR_DIR}/AddMLIR.cmake)
