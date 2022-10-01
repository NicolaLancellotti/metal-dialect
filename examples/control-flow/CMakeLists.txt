add_llvm_executable(examplesControlFlow
        main.cpp
)

target_link_libraries(examplesControlFlow
        MLIRExecutionEngine
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRTranslateLib
        MLIRSPIRVDialect
        MLIRSupport
        LLVMSupport
        MLIRMetal
        examplesUtility
)
