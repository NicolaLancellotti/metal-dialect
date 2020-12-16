add_llvm_executable(examplesMemory
        main.cpp
)

target_link_libraries(examplesMemory
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

