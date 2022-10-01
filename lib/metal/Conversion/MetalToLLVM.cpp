//===--- MetalToLLVM.cpp --------------------------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal/Conversion/MetalToLLVM.h"
#include "metal/IR/MetalOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

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
      loc, LLVM::LLVMPointerType::get(builder.getIntegerType(8)), globalPtr,
      ArrayRef<Value>({cst0, cst0}));
}

static void rewriteOp(Operation *op, ArrayRef<Value> operands,
                      ConversionPatternRewriter &rewriter,
                      llvm::StringRef functionName,
                      llvm::Optional<Type> resultType, ArrayRef<Type> params) {
  auto loc = op->getLoc();
  auto module = op->getParentOfType<ModuleOp>();
  auto *context = module.getContext();

  SymbolRefAttr callee;
  if (module.lookupSymbol<LLVMFuncOp>(functionName))
    callee = SymbolRefAttr::get(context, functionName);
  else {
    if (resultType.hasValue()) {
      auto llvmFnType =
          LLVM::LLVMFunctionType::get(resultType.getValue(), params, false);
      callee = insertFunction(rewriter, module, llvmFnType, functionName);
    } else {
      auto llvmVoidTy = LLVM::LLVMVoidType::get(context);
      auto llvmFnType = LLVM::LLVMFunctionType::get(llvmVoidTy, params, false);
      callee = insertFunction(rewriter, module, llvmFnType, functionName);
    }
  }
  if (resultType.hasValue()) {
    auto call = rewriter.create<func::CallOp>(loc, callee,
                                              resultType.getValue(), operands);
    rewriter.replaceOp(op, call.getResult(0));
  } else {
    rewriter.create<func::CallOp>(loc, callee, llvm::None, operands);
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
    auto i32PtrTy = LLVM::LLVMPointerType::get(rewriter.getI32Type());
    auto i64Ty = rewriter.getI64Type();
    auto arrayTy = LLVM::LLVMArrayType::get(i64Ty, 1);
    auto structTy = LLVM::LLVMStructType::getLiteral(
        context, {i32PtrTy, i32PtrTy, i64Ty, arrayTy, arrayTy});
    auto structPtrTy = LLVM::LLVMPointerType::get(structTy);

    SymbolRefAttr callee;
    if (module.lookupSymbol<LLVMFuncOp>(functionName))
      callee = SymbolRefAttr::get(context, functionName);
    else {
      ArrayRef<Type> types = {i64Ty, structPtrTy};
      auto llvmFnType = LLVM::LLVMFunctionType::get(voidTy, types, false);
      callee = insertFunction(rewriter, module, llvmFnType, functionName);
    }

    auto one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(1));
    auto alloca =
        rewriter.create<mlir::LLVM::AllocaOp>(loc, structPtrTy, one, 0);

    ArrayRef<Value> newOperand = {operands[0], alloca};
    rewriter.create<func::CallOp>(loc, callee, llvm::None, newOperand);
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
    auto i64Ty = rewriter.getI64Type();
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getIntegerType(8));

    auto lib =
        getOrCreateGlobalString(loc, rewriter, "metallib",
                                StringRef("./default.metallib\0", 19), module);

    auto createOp = cast<mlir::metal::CommandQueueMakeCommandBufferOp>(op);
    auto kernel =
        getOrCreateGlobalString(loc, rewriter, createOp.functionName(),
                                createOp.functionName(), module);

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
    auto i64Ty = rewriter.getI64Type();
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
    auto i64Ty = rewriter.getI64Type();
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
    auto i64Ty = rewriter.getI64Type();
    rewriteOp(op, operands, rewriter, "_MetalCommandBufferWaitUntilCompleted",
              llvm::None, {i64Ty});
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
