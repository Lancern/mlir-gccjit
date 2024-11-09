include_guard(GLOBAL)

set(GCCJIT_ROOT "" CACHE PATH "Path to GCCJIT installation")
set(GCCJIT_INCLUDE_DIR "" CACHE PATH "Path to GCCJIT include directory")

if (GCCJIT_ROOT)
    message(STATUS "Using GCCJIT from ${GCCJIT_ROOT}")

    if (NOT GCCJIT_INCLUDE_DIR)
        set(GCCJIT_INCLUDE_DIR ${GCCJIT_ROOT}/include)
    endif ()

    set(GCCJIT_LIB_DIRS ${GCCJIT_ROOT})
    if (EXISTS ${GCCJIT_ROOT}/lib)
        list(APPEND GCCJIT_LIB_DIRS ${GCCJIT_ROOT}/lib)
    endif()
    if (EXISTS ${GCCJIT_ROOT}/lib/gcc/current)
        list(APPEND GCCJIT_LIB_DIRS ${GCCJIT_ROOT}/lib/gcc/current)
    endif ()
endif ()

if (GCCJIT_INCLUDE_DIR)
    list(APPEND CMAKE_REQUIRED_INCLUDES ${GCCJIT_INCLUDE_DIR})
endif ()

include(CheckIncludeFileCXX)
CHECK_INCLUDE_FILE_CXX(libgccjit.h LIBGCCJIT_H_EXIST)
if (NOT LIBGCCJIT_H_EXIST)
    message(FATAL_ERROR "could not find libgccjit.h in system headers (CMAKE_REQUIRED_INCLUDES: ${CMAKE_REQUIRED_INCLUDES})")
endif ()

if (GCCJIT_LIB_DIRS OR GCCJIT_LIBRARY)
    find_library(GCCJIT_LIBRARY NAMES gccjit PATHS ${GCCJIT_LIB_DIRS})
    if (NOT GCCJIT_LIBRARY)
        message(FATAL_ERROR "could not find gccjit library file at ${GCCJIT_LIB_DIRS}")
    elseif (NOT EXISTS ${GCCJIT_LIBRARY})
        message(FATAL_ERROR "could not find gccjit library file at ${GCCJIT_LIBRARY}")
    endif()

    add_library(libgccjit SHARED IMPORTED)
    set_target_properties(libgccjit PROPERTIES
        IMPORTED_LOCATION ${GCCJIT_LIBRARY}
    )
else ()
    add_library(libgccjit INTERFACE)
    target_link_libraries(libgccjit INTERFACE gccjit)
endif ()

if (GCCJIT_INCLUDE_DIR)
    target_include_directories(libgccjit INTERFACE ${GCCJIT_INCLUDE_DIR})
endif ()
