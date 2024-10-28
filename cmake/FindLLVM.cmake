include_guard(GLOBAL)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")

include(${LLVM_DIR}/AddLLVM.cmake)
include(${LLVM_DIR}/TableGen.cmake)
include(${LLVM_DIR}/HandleLLVMOptions.cmake)

# Try to find the FileCheck utility.
if (TARGET FileCheck)
    get_target_property(LLVM_FILE_CHECK_EXE FileCheck IMPORTED_LOCATION)
endif ()

if (NOT LLVM_FILE_CHECK_EXE)
    unset(LLVM_FILE_CHECK_EXE)
    if (MLIR_GCCJIT_ENABLE_TEST)
        set(filecheck_required_param REQUIRED)
    endif ()
    find_program(LLVM_FILE_CHECK_EXE
        FileCheck
        NAMES FileCheck-${LLVM_VERSION_MAJOR}
        HINTS
            "${LLVM_DIR}/../../../bin"
            "${LLVM_DIR}/../bin"
        ${filecheck_required_param}
    )
endif ()

if (LLVM_FILE_CHECK_EXE)
    message(STATUS "Found FileCheck utility at ${LLVM_FILE_CHECK_EXE}")
elseif (MLIR_GCCJIT_ENABLE_TEST)
    message(FATAL_ERROR "Could not find FileCheck utility")
endif ()
