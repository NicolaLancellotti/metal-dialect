//===--- MetalToLLVM.cpp --------------------------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal/Conversion/MetalToLLVM.h"
#include "metal/IR/MetalOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace {

using namespace mlir;
using namespace LLVM;

static SymbolRefAttr insertFunction(ConversionPatternRewriter &rewriter,
                                    ModuleOp module,
                                    LLVM::LLVMFunctionType llvmFnType,
                                    llvm::StringRef functionName) {
  auto *context = module.getContext();
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), functionName, llvmFnType);
  return SymbolRefAttr::get(context, functionName);
}

static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                     StringRef name, StringRef value,
                                     ModuleOp module) {
  LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type =
        LLVM::LLVMArrayType::get(builder.getIntegerType(8), value.size());
    global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                            LLVM::Linkage::Internal, name,
                                            builder.getStringAttr(value));
  }

  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value cst0 = builder.create<LLVM::ConstantOp>(
      loc, builder.getI64Type(),
      builder.getIntegerAttr(builder.getIndexType(), 0));
  return builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getType(),
      globalPtr, ArrayRef<Value>({cst0, cst0}));
}

static void rewriteOp(Operation *op, ArrayRef<Value> operands,
                      ConversionPatternRewriter &rewriter,
                      llvm::StringRef functionName,
                      std::optional<Type> resultType, ArrayRef<Type> params) {
  auto loc = op->getLoc();
  auto module = op->getParentOfType<ModuleOp>();
  auto *context = module.getContext();

  SymbolRefAttr callee;
  if (module.lookupSymbol<LLVMFuncOp>(functionName))
    callee = SymbolRefAttr::get(context, functionName);
  else {
    if (resultType.has_value()) {
      auto llvmFnType =
          LLVM::LLVMFunctionType::get(resultType.value(), params, false);
      callee = insertFunction(rewriter, module, llvmFnType, functionName);
    } else {
      auto llvmVoidTy = LLVM::LLVMVoidType::get(context);
      auto llvmFnType = LLVM::LLVMFunctionType::get(llvmVoidTy, params, false);
      callee = insertFunction(rewriter, module, llvmFnType, functionName);
    }
  }
  if (resultType.has_value()) {
    auto call = rewriter.create<func::CallOp>(loc, callee, resultType.value(),
                                              operands);
    rewriter.replaceOp(op, call.getResult(0));
  } else {
    rewriter.create<func::CallOp>(loc, callee, std::nullopt, operands);
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
    auto i64Ty = rewriter.getI64Type();
    rewriteOp(op, operands, rewriter, "_MetalRelease", std::nullopt, {i64Ty});
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
    auto i64Ty = rewriter.getI64Type();
    rewriteOp(op, operands, rewriter, "_MetalDeviceMakeDefault", i64Ty, {});
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
    auto i64Ty = rewriter.getI64Type();
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
    auto i64Ty = rewriter.getI64Type();
    auto i1Ty = rewriter.getI1Type();
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
    auto module = op->getParentOfType<ModuleOp>();
    auto *context = module.getContext();
    auto functionName = "_MetalBufferGetContents";

    auto voidTy = LLVM::LLVMVoidType::get(context);
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64Ty = rewriter.getI64Type();
    auto arrayTy = LLVM::LLVMArrayType::get(i64Ty, 1);
    auto structTy = LLVM::LLVMStructType::getLiteral(
        context, {ptrTy, ptrTy, i64Ty, arrayTy, arrayTy});

    SymbolRefAttr callee;
    if (module.lookupSymbol<LLVMFuncOp>(functionName))
      callee = SymbolRefAttr::get(context, functionName);
    else {
      ArrayRef<Type> types = {i64Ty, ptrTy};
      auto llvmFnType = LLVM::LLVMFunctionType::get(voidTy, types, false);
      callee = insertFunction(rewriter, module, llvmFnType, functionName);
    }

    auto one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(1));
    auto alloca =
        rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrTy, structTy, one, 0);

    ArrayRef<Value> newOperand = {operands[0], alloca};
    rewriter.create<func::CallOp>(loc, callee, std::nullopt, newOperand);
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, structTy, alloca);
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
    auto i64Ty = rewriter.getI64Type();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    auto lib =
        getOrCreateGlobalString(loc, rewriter, "metallib",
                                StringRef("./default.metallib\0", 19), module);

    auto createOp = cast<mlir::metal::CommandQueueMakeCommandBufferOp>(op);
    auto kernel =
        getOrCreateGlobalString(loc, rewriter, createOp.getFunctionName(),
                                createOp.getFunctionName(), module);

    ArrayRef<Value> newOperands = {operands[0], lib,         kernel,
                                   operands[1], operands[2], operands[3]};

    rewriteOp(op, newOperands, rewriter, "_MetalCommandQueueMakeCommandBuffer",
              i64Ty, {i64Ty, ptrTy, ptrTy, i64Ty, i64Ty, i64Ty});
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
    auto i64Ty = rewriter.getI64Type();
    rewriteOp(op, operands, rewriter, "_MetalCommandBufferAddBuffer",
              std::nullopt, {i64Ty, i64Ty, i64Ty});
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
    auto i64Ty = rewriter.getI64Type();
    rewriteOp(op, operands, rewriter, "_MetalCommandBufferCommit", std::nullopt,
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
    auto i64Ty = rewriter.getI64Type();
    rewriteOp(op, operands, rewriter, "_MetalCommandBufferWaitUntilCompleted",
              std::nullopt, {i64Ty});
    return success();
  }
};

} // end namespace

void mlir::metal::populateMetalToLLVMConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<
      ModuleOpLowering, ReleaseOpLowering, DeviceMakeDefaultOpLowering,
      DeviceMakeCommandQueueOpLowering, DeviceMakeBufferOpLowering,
      BufferGetContentsOpLowering, CommandQueueMakeCommandBufferOpLowering,
      CommandBufferAddBufferOpLowering, CommandBufferCommitOpLowering,
      CommandBufferWaitUntilCompletedOpLowering>(ctx);
}
