set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name MetalConversion)
add_public_tablegen_target(MLIRMetalConversionPassIncGen)

add_mlir_doc(Passes MetalConversionPasses ./metal/ -gen-pass-doc)
