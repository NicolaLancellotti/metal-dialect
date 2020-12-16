add_llvm_executable(examplesTheGameOfLife
        main.cpp
)

target_link_libraries(examplesTheGameOfLife
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

