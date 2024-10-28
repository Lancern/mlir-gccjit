configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

add_custom_target(check
    COMMAND "${PROJECT_SOURCE_DIR}/run-lit.py" -v ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS gccjit-tools
)
