include_guard(GLOBAL)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")

include(${LLVM_DIR}/AddLLVM.cmake)
include(${LLVM_DIR}/TableGen.cmake)
include(${LLVM_DIR}/HandleLLVMOptions.cmake)

# Try to find the FileCheck utility.
if (NOT TARGET FileCheck)
    find_program(LLVM_FILE_CHECK_EXE
        FileCheck
        NAMES FileCheck-${LLVM_VERSION_MAJOR}
        HINTS "${LLVM_DIR}/../../../bin"
    )
    if (LLVM_FILE_CHECK_EXE)
        add_executable(FileCheck IMPORTED)
        set_target_properties(FileCheck PROPERTIES IMPORTED_LOCATION ${LLVM_FILE_CHECK_EXE})
    endif ()
endif ()

if (TARGET FileCheck)
    if (NOT LLVM_FILE_CHECK_EXE)
        get_target_property(LLVM_FILE_CHECK_EXE FileCheck IMPORTED_LOCATION)
    endif ()
    message(STATUS "Found FileCheck utility at ${LLVM_FILE_CHECK_EXE}")
elseif (MLIR_GCCJIT_ENABLE_TEST)
    message(FATAL_ERROR "Could not find FileCheck utility")
endif ()
