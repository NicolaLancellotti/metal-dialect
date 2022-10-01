add_llvm_executable(examplesMemory
        main.cpp
)

target_link_libraries(examplesMemory
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
