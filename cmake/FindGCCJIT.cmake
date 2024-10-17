include_guard(GLOBAL)

include(CheckIncludeFileCXX)
CHECK_INCLUDE_FILE_CXX(libgccjit.h LIBGCCJIT_H_EXIST)
if (NOT LIBGCCJIT_H_EXIST)
    message(FATAL_ERROR "could not find libgccjit.h in system headers")
endif ()

add_library(libgccjit INTERFACE)
target_link_libraries(libgccjit INTERFACE gccjit)
