cmake_minimum_required(VERSION 3.22)
project(mlir-gccjit C CXX)

if (NOT PROJECT_IS_TOP_LEVEL)
    message(FATAL_ERROR "For now mlir-gccjit must be the root project during build")
endif ()

option(MLIR_GCCJIT_BUILD_TOOLS "Build mlir-gccjit tools" ON)
option(MLIR_GCCJIT_ENABLE_TEST "Enable test targets in build" ON)

set(LLVM_BUILD_TOOLS ${MLIR_GCCJIT_BUILD_TOOLS})

include(cmake/FindLLVM.cmake)
include(cmake/FindMLIR.cmake)
include(cmake/FindGCCJIT.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(src)

add_subdirectory(tools)

if (MLIR_GCCJIT_ENABLE_TEST)
    add_subdirectory(test)
endif ()
