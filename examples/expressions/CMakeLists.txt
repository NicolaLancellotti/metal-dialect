add_llvm_executable(examplesExpressions
        main.cpp
)

target_link_libraries(examplesExpressions
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

