set(LLVM_TARGET_DEFINITIONS MetalPasses.td)
mlir_tablegen(MetalPasses.h.inc -gen-pass-decls -name MetalConversion)
add_public_tablegen_target(MLIRMetalConversionPassIncGen)

add_mlir_doc(MetalPasses MetalConversionPasses ./metal/ -gen-pass-doc)