//===--- LowerToLLVMPass.cpp-----------------------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "Metal/Conversion/MetalToLLVM.h"
#include "Metal/Dialect/MetalOps.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct LLVMLoweringPass
    : public PassWrapper<LLVMLoweringPass, OperationPass<ModuleOp>> {

  void runOnOperation() final {
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

    OwningRewritePatternList patterns;
    LLVMTypeConverter typeConverter(&getContext());
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    populateMetalToLLVMConversionPatterns(patterns, &getContext());

    if (failed(applyFullConversion(getOperation(), target, patterns)))
      signalPassFailure();
  }
};

} // end namespace

namespace mlir {

std::unique_ptr<mlir::Pass> createLowerMetalToLLVMPass() {
  return std::make_unique<LLVMLoweringPass>();
}

void registerLowerMetaToLLVMPass() {
  PassRegistration<LLVMLoweringPass>(
      "lower-metal-to-llvm", "Lower Metal operations into the LLVM dialect");
}

} // end namespace mlir
