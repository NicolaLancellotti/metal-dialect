add_mlir_dialect_library(MLIRMetal
		Conversion/MetalToLLVM.cpp
		Conversion/LowerToLLVMPass.cpp

		Dialect/MetalDialect.cpp
		Dialect/MetalOps.cpp

		Target/ConvertToMetalShadingLanguage.cpp
		Target/ModuleTranslation.cpp

		ADDITIONAL_HEADER_DIRS
		${PROJECT_SOURCE_DIR}/include/Metal

		DEPENDS
		MLIRMetalOpsIncGen

		LINK_LIBS PUBLIC
		MLIRIR
)
