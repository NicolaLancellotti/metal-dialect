add_llvm_executable(examplesControlFlow
        main.cpp
)

target_link_libraries(examplesControlFlow
        MLIRExecutionEngine
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRTranslation
        MLIRSPIRV
        MLIRStandardToLLVM
        MLIRSCFToStandard
        MLIRSupport
        LLVMSupport
        MLIRMetal
        examplesUtility
)

