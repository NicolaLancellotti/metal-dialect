add_llvm_executable(examplesFold
        main.cpp
)

target_link_libraries(examplesFold
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

