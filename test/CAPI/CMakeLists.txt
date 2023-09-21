# Here we create a single aggregate shared library with the parts of the CAPI
# that we want to bundle together. Then we link a simple C executable
# against it to demonstrate that it does have the fully self contained
# core MLIR library and our own metal dialect.
add_mlir_aggregate(MetalCAPITestLib
  SHARED
  EMBED_LIBS
  MLIRCAPIIR
  # TODO: Remove this in favor of showing fine grained dialect registration
  # (once available).
  MLIRCAPIRegisterEverything
  MetalCAPI
)

add_llvm_executable(metal-capi-test
  metal-capi-test.c
)
llvm_update_compile_flags(metal-capi-test)
target_link_libraries(metal-capi-test
  PRIVATE MetalCAPITestLib)
