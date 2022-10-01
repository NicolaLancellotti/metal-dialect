add_llvm_executable(examplesTheGameOfLife
        main.cpp
)

target_link_libraries(examplesTheGameOfLife
        MLIRExecutionEngine
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRTranslateLib
        MLIRSPIRVDialect
        MLIRMemRefDialect
        MLIRSupport
        LLVMSupport
        MLIRMetal
        examplesUtility
)
