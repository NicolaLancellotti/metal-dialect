add_mlir_public_c_api_library(MetalCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  MLIRMetal
)
