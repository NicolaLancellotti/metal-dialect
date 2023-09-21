set(LIB_HEADERS  ${PROJECT_SOURCE_DIR}/include/metal)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_dialect_library(MLIRMetal
		# Conversion
		${LIB_HEADERS}/Conversion/MetalPasses.h
		${LIB_HEADERS}/Conversion/MetalToLLVM.h
		Conversion/ConvertMetalToLLVM.cpp
		Conversion/MetalToLLVM.cpp

		# IR
		${LIB_HEADERS}/IR/MetalDialect.h
		IR/MetalDialect.cpp
		${LIB_HEADERS}/IR/MetalOps.h
		IR/MetalOps.cpp
		${LIB_HEADERS}/IR/MetalTypes.h
		IR/MetalTypes.cpp

		# Target
		${LIB_HEADERS}/Target/MetalShadingLanguage.h
		Target/ConvertToMetalShadingLanguage.cpp
		${LIB_HEADERS}/Target/ModuleTranslation.h
		Target/ModuleTranslation.cpp

		DEPENDS
		MLIRMetalOpsIncGen
		MLIRMetalOpsEnumsGen
		MLIRMetalConversionPassIncGen

		LINK_LIBS PUBLIC
		MLIRIR
		${conversion_libs}
)
