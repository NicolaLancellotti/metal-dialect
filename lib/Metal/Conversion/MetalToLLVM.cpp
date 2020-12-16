//===--- MetalToLLVM.cpp --------------------------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "Metal/Conversion/MetalToLLVM.h"
#include "Metal/Dialect/MetalOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

using namespace LLVM;

static LLVMDialect *getLLVMDialect(Operation *op) {
  return op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
}

static SymbolRefAttr insertFunction(ConversionPatternRewriter &rewriter,
                                    ModuleOp module, LLVMType llvmFnType,
                                    llvm::StringRef functionName) {
  auto *context = module.getContext();
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), functionName, llvmFnType);
  return SymbolRefAttr::get(functionName, context);
}

static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                     StringRef name, StringRef value,
                                     ModuleOp module,
                                     LLVM::LLVMDialect *llvmDialect) {
  LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMType::getArrayTy(
        LLVM::LLVMType::getInt8Ty(llvmDialect), value.size());
    global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                            LLVM::Linkage::Internal, name,
                                            builder.getStringAttr(value));
  }

  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value cst0 = builder.create<LLVM::ConstantOp>(
      loc, LLVM::LLVMType::getInt64Ty(llvmDialect),
      builder.getIntegerAttr(builder.getIndexType(), 0));
  return builder.create<LLVM::GEPOp>(loc,
                                     LLVM::LLVMType::getInt8PtrTy(llvmDialect),
                                     globalPtr, ArrayRef<Value>({cst0, cst0}));
}

static void rewriteOp(Operation *op, ArrayRef<Value> operands,
                      ConversionPatternRewriter &rewriter,
                      llvm::StringRef functionName,
                      llvm::Optional<LLVMType> resultType,
                      ArrayRef<LLVMType> params) {
  auto loc = op->getLoc();
  auto module = op->getParentOfType<ModuleOp>();
  auto *context = module.getContext();

  SymbolRefAttr callee;
  if (module.lookupSymbol<LLVMFuncOp>(functionName))
    callee = SymbolRefAttr::get(functionName, context);
  else {
    if (resultType.hasValue()) {
      auto llvmFnType =
          LLVMType::getFunctionTy(resultType.getValue(), params, false);
      callee = insertFunction(rewriter, module, llvmFnType, functionName);
    } else {
      auto *llvmDialect = getLLVMDialect(op);
      auto llvmVoidTy = LLVMType::getVoidTy(llvmDialect);
      auto llvmFnType = LLVMType::getFunctionTy(llvmVoidTy, params, false);
      callee = insertFunction(rewriter, module, llvmFnType, functionName);
    }
  }
  if (resultType.hasValue()) {
    auto call = rewriter.create<mlir::CallOp>(loc, callee,
                                              resultType.getValue(), operands);
    rewriter.replaceOp(op, call.getResult(0));
  } else {
    rewriter.create<mlir::CallOp>(loc, callee, llvm::None, operands);
    rewriter.eraseOp(op);
  }
}

class ModuleOpLowering : public ConversionPattern {
public:
  explicit ModuleOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::metal::ModuleOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

class ReleaseOpLowering : public ConversionPattern {
public:
  explicit ReleaseOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::metal::ReleaseOp::getOperationName(), 1,
                          context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto *llvmDialect = getLLVMDialect(op);
    auto i64Ty = LLVMType::getInt64Ty(llvmDialect);
    rewriteOp(op, operands, rewriter, "_MetalRelease", llvm::None, {i64Ty});
    return success();
  }
};

class DeviceMakeDefaultOpLowering : public ConversionPattern {
public:
  explicit DeviceMakeDefaultOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::metal::DeviceMakeDefaultOp::getOperationName(),
                          1, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto *llvmDialect = getLLVMDialect(op);
    auto resultType = LLVMType::getInt64Ty(llvmDialect);
    rewriteOp(op, operands, rewriter, "_MetalDeviceMakeDefault", resultType,
              {});
    return success();
  }
};

class DeviceMakeCommandQueueOpLowering : public ConversionPattern {
public:
  explicit DeviceMakeCommandQueueOpLowering(MLIRContext *context)
      : ConversionPattern(
            mlir::metal::DeviceMakeCommandQueueOp::getOperationName(), 1,
            context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto *llvmDialect = getLLVMDialect(op);
    auto i64Ty = LLVMType::getInt64Ty(llvmDialect);
    rewriteOp(op, operands, rewriter, "_MetalDeviceMakeCommandQueue", i64Ty,
              {i64Ty});
    return success();
  }
};

class DeviceMakeBufferOpLowering : public ConversionPattern {
public:
  explicit DeviceMakeBufferOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::metal::DeviceMakeBufferOp::getOperationName(),
                          1, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto *llvmDialect = getLLVMDialect(op);
    auto i64Ty = LLVMType::getInt64Ty(llvmDialect);
    auto i1Ty = LLVMType::getInt1Ty(llvmDialect);
    rewriteOp(op, operands, rewriter, "_MetalDeviceMakeBuffer", i64Ty,
              {i64Ty, i1Ty, i64Ty, i64Ty});
    return success();
  }
};

class BufferGetContentsOpLowering : public ConversionPattern {
public:
  explicit BufferGetContentsOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::metal::BufferGetContentsOp::getOperationName(),
                          1, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto *llvmDialect = getLLVMDialect(op);
    auto module = op->getParentOfType<ModuleOp>();
    auto *context = module.getContext();
    auto functionName = "_MetalBufferGetContents";

    auto llvmVoidTy = LLVMType::getVoidTy(llvmDialect);
    auto llvmI32PtrTy = LLVMType::getInt32Ty(llvmDialect).getPointerTo();
    auto llvmI64Ty = LLVMType::getInt64Ty(llvmDialect);
    auto arrayTy = LLVMType::getArrayTy(llvmI64Ty, 1);
    auto structType = LLVMType::getStructTy(
        llvmDialect, {llvmI32PtrTy, llvmI32PtrTy, llvmI64Ty, arrayTy, arrayTy});

    SymbolRefAttr callee;
    if (module.lookupSymbol<LLVMFuncOp>(functionName))
      callee = SymbolRefAttr::get(functionName, context);
    else {
      ArrayRef<LLVMType> types = {llvmI64Ty, structType.getPointerTo()};
      auto llvmFnType = LLVMType::getFunctionTy(llvmVoidTy, types, false);
      callee = insertFunction(rewriter, module, llvmFnType, functionName);
    }

    auto one =
        rewriter.create<mlir::ConstantOp>(loc, rewriter.getI32IntegerAttr(1));
    auto alloca = rewriter.create<mlir::LLVM::AllocaOp>(
        loc, structType.getPointerTo(), one, 0);

    ArrayRef<Value> newOperand = {operands[0], alloca};
    rewriter.create<mlir::CallOp>(loc, callee, llvm::None, newOperand);
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, alloca);
    return success();
  }
};

class CommandQueueMakeCommandBufferOpLowering : public ConversionPattern {
public:
  explicit CommandQueueMakeCommandBufferOpLowering(MLIRContext *context)
      : ConversionPattern(
            mlir::metal::CommandQueueMakeCommandBufferOp::getOperationName(), 1,
            context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto *llvmDialect = getLLVMDialect(op);
    auto i64Ty = LLVMType::getInt64Ty(llvmDialect);
    auto i8PtrTy = LLVM::LLVMType::getInt8PtrTy(llvmDialect);

    auto lib = getOrCreateGlobalString(loc, rewriter, "metallib",
                                       StringRef("./default.metallib\0", 19),
                                       module, llvmDialect);

    auto createOp = cast<mlir::metal::CommandQueueMakeCommandBufferOp>(op);
    auto kernel =
        getOrCreateGlobalString(loc, rewriter, createOp.functionName(),
                                createOp.functionName(), module, llvmDialect);

    ArrayRef<Value> newOperands = {operands[0], lib,         kernel,
                                   operands[1], operands[2], operands[3]};

    rewriteOp(op, newOperands, rewriter, "_MetalCommandQueueMakeCommandBuffer",
              i64Ty, {i64Ty, i8PtrTy, i8PtrTy, i64Ty, i64Ty, i64Ty});
    return success();
  }
};

class CommandBufferAddBufferOpLowering : public ConversionPattern {
public:
  explicit CommandBufferAddBufferOpLowering(MLIRContext *context)
      : ConversionPattern(
            mlir::metal::CommandBufferAddBufferOp::getOperationName(), 1,
            context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto *llvmDialect = getLLVMDialect(op);
    auto i64Ty = LLVMType::getInt64Ty(llvmDialect);
    rewriteOp(op, operands, rewriter, "_MetalCommandBufferAddBuffer",
              llvm::None, {i64Ty, i64Ty, i64Ty});
    return success();
  }
};

class CommandBufferCommitOpLowering : public ConversionPattern {
public:
  explicit CommandBufferCommitOpLowering(MLIRContext *context)
      : ConversionPattern(
            mlir::metal::CommandBufferCommitOp::getOperationName(), 1,
            context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto *llvmDialect = getLLVMDialect(op);
    auto i64Ty = LLVMType::getInt64Ty(llvmDialect);
    rewriteOp(op, operands, rewriter, "_MetalCommandBufferCommit", llvm::None,
              {i64Ty});
    return success();
  }
};

class CommandBufferWaitUntilCompletedOpLowering : public ConversionPattern {
public:
  explicit CommandBufferWaitUntilCompletedOpLowering(MLIRContext *context)
      : ConversionPattern(
            mlir::metal::CommandBufferWaitUntilCompletedOp::getOperationName(),
            1, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto *llvmDialect = getLLVMDialect(op);
    auto i64Ty = LLVMType::getInt64Ty(llvmDialect);
    rewriteOp(op, operands, rewriter, "_MetalCommandBufferWaitUntilCompleted",
              llvm::None, {i64Ty});
    return success();
  }
};

} // end namespace

void mlir::populateMetalToLLVMConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<
      ModuleOpLowering, ReleaseOpLowering, DeviceMakeDefaultOpLowering,
      DeviceMakeCommandQueueOpLowering, DeviceMakeBufferOpLowering,
      BufferGetContentsOpLowering, CommandQueueMakeCommandBufferOpLowering,
      CommandBufferAddBufferOpLowering, CommandBufferCommitOpLowering,
      CommandBufferWaitUntilCompletedOpLowering>(ctx);
}
