set(dialect MetalOps)
set(dialect_namespace metal)

set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
mlir_tablegen(${dialect}.h.inc -gen-op-decls)
mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
mlir_tablegen(${dialect}Enums.h.inc -gen-enum-decls)
mlir_tablegen(${dialect}Enums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIR${dialect}IncGen)
add_dependencies(mlir-headers MLIR${dialect}IncGen)

add_mlir_doc(MetalDialect -gen-dialect-doc MetalDialect Metal/)
add_mlir_doc(MetalOps -gen-op-doc MetalOps Metal/)