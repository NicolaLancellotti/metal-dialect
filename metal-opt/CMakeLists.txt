get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRMetal
        )
add_llvm_executable(metal-opt metal-opt.cpp)

llvm_update_compile_flags(metal-opt)
target_link_libraries(metal-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(metal-opt)
