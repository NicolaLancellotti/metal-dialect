//===--- ConvertMetalToLLVM.cpp--------------------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal/Conversion/MetalToLLVM.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "PassDetail.h"

using namespace mlir;

namespace {

struct ConvertMetalToLLVM : public ConvertMetalToLLVMBase<ConvertMetalToLLVM> {
  ConvertMetalToLLVM() = default;

  void runOnOperation() final {
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();

    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter typeConverter(&getContext());
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter,
                                                            patterns);
    mlir::metal::populateMetalToLLVMConversionPatterns(patterns, &getContext());

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // end namespace

namespace mlir {

std::unique_ptr<mlir::Pass> mlir::metal::createConvertMetalToLLVMPass() {
  return std::make_unique<ConvertMetalToLLVM>();
}

} // end namespace mlir
