configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(METAL_TEST_DEPENDS
        FileCheck count not
        metal-opt
        metal-translate
        )

add_lit_testsuite(check-metal "Running the metal regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${METAL_TEST_DEPENDS}
        )
set_target_properties(check-metal PROPERTIES FOLDER "Tests")

add_lit_testsuites(METAL ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${METAL_TEST_DEPENDS})
