set(LIB_HEADERS  ${PROJECT_SOURCE_DIR}/include/metal)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_dialect_library(MLIRMetal
		${LIB_HEADERS}/Conversion/MetalToLLVM.h
		Conversion/MetalToLLVM.cpp
		${LIB_HEADERS}/Conversion/Passes.h
		Conversion/ConvertMetalToLLVM.cpp
		Conversion/PassDetail.h

		${LIB_HEADERS}/IR/MetalDialect.h
		IR/MetalDialect.cpp
		${LIB_HEADERS}/IR/MetalMemRefType.h
		${LIB_HEADERS}/IR/MetalOps.h
		IR/MetalOps.cpp

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
