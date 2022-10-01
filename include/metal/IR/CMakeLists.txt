add_mlir_dialect(MetalOps metal)

set(LLVM_TARGET_DEFINITIONS MetalOps.td)
mlir_tablegen(MetalOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(MetalOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRMetalOpsEnumsGen)
