//===--- LowerToLLVMPass.cpp-----------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Metal/Conversion/MetalToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct LLVMLoweringPass
    : public PassWrapper<LLVMLoweringPass, OperationPass<ModuleOp>> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() final {
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

    OwningRewritePatternList patterns;
    LLVMTypeConverter typeConverter(&getContext());
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    populateMetalToLLVMConversionPatterns(patterns, &getContext());

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
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
