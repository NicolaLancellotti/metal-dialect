//===--- ConvertMetalToLLVM.cpp--------------------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal/Conversion/MetalPasses.h"
#include "metal/Conversion/MetalToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::metal {

#define GEN_PASS_DEF_CONVERTMETALTOLLVM
#include "metal/Conversion/MetalPasses.h.inc"

namespace {
struct ConvertMetalToLLVM
    : public impl::ConvertMetalToLLVMBase<ConvertMetalToLLVM> {

  using impl::ConvertMetalToLLVMBase<
      ConvertMetalToLLVM>::ConvertMetalToLLVMBase;

  void runOnOperation() final {
    LLVMConversionTarget target(getContext());
    target.addLegalOp<mlir::ModuleOp>();

    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter typeConverter(&getContext());
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::metal::populateMetalToLLVMConversionPatterns(patterns, &getContext());

    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyFullConversion(getOperation(), target, patternSet)))
      signalPassFailure();
  }
};

} // end namespace
} // namespace mlir::metal
