#include "Metal/Conversion/MetalToLLVM.h"
#include "Metal/Dialect/MetalDialect.h"
#include "Metal/Dialect/MetalMemRefType.h"
#include "Metal/Dialect/MetalOps.h"
#include "Metal/Target/MetalShadingLanguage.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#ifndef METAL_DRIVER_H
#define METAL_DRIVER_H

class Driver {
public:
  Driver() {
    mlir::registerDialect<mlir::metal::MetalDialect>();
    mlir::registerDialect<mlir::StandardOpsDialect>();
    mlir::registerDialect<mlir::LLVM::LLVMDialect>();
    this->_context = std::make_unique<mlir::MLIRContext>();
    this->_pm = std::make_unique<mlir::PassManager>(&*_context);
    this->_builder = std::make_unique<mlir::OpBuilder>(&*_context);
    this->_loc = std::make_unique<mlir::Location>(_builder->getFileLineColLoc(
        mlir::Identifier::get("0", _builder->getContext()), 0, 0));

    this->_module = std::make_unique<mlir::ModuleOp>(
        mlir::ModuleOp::create(_builder->getUnknownLoc()));
    this->_metalModule = std::make_unique<mlir::metal::ModuleOp>(
        _builder->create<mlir::metal::ModuleOp>(*_loc));
    _module->push_back(*_metalModule);
  }

  void dump() { _module->dump(); }

  mlir::OpBuilder &builder() { return *_builder; }

  mlir::Location &loc() { return *_loc; }

  void addKernel(mlir::metal::KernelOp op) { _metalModule->push_back(op); }

  void addOperation(mlir::Operation *op) { _module->push_back(op); }

  void verify() {
    if (failed(mlir::verify(*_module))) {
      _module->emitError("module verification error");
      exit(EXIT_FAILURE);
    }
  }

  void canonicalize() {
    mlir::OpPassManager &optPM = _pm->nest<mlir::metal::ModuleOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    if (mlir::failed(_pm->run(*_module))) {
      exit(EXIT_FAILURE);
    }
  }

  void dumpMSL() {
    llvm::errs() << "\n";
    if (mlir::failed(mlir::metal::translateModuleToMetalShadingLanguage(
            *_module, llvm::errs())))
      exit(EXIT_FAILURE);
  }

  void translateToMSL() {
    llvm::StringRef metalFilePath = "./default.metal";
    llvm::StringRef metalLibPath = "./default.metallib";
    std::error_code errorCode;
    llvm::raw_fd_ostream stream(metalFilePath, errorCode,
                                llvm::sys::fs::FileAccess::FA_Write);
    if (errorCode) {
      llvm::errs() << "Could not open file: " << errorCode.message();
      exit(EXIT_FAILURE);
    }

    if (mlir::failed(mlir::metal::translateModuleToMetalShadingLanguage(
            *_module, stream)))
      exit(EXIT_FAILURE);

    auto command = llvm::Twine("xcrun -sdk macosx metal ") + metalFilePath +
                   " -o " + metalLibPath;
    if (system(command.str().c_str())) {
      llvm::errs() << "error";
      exit(EXIT_FAILURE);
    }
  }

  void translateToLLVM() {
    _pm->addPass(mlir::createLowerMetalToLLVMPass());
    if (mlir::failed(_pm->run(*_module)))
      exit(EXIT_FAILURE);

    auto llvmModule = mlir::translateModuleToLLVMIR(*_module);
    if (!llvmModule)
      exit(EXIT_FAILURE);

    llvm::StringRef objectFilePath = "./main.o";
    genObjectFile(llvmModule, objectFilePath);
  }

  mlir::LLVM::LLVMFuncOp insertPutchar() {
    auto llvmDialect =
        _builder->getContext()->getRegisteredDialect<mlir::LLVM::LLVMDialect>();
    auto llvmI32Ty = mlir::LLVM::LLVMType::getInt64Ty(llvmDialect);
    auto llvmFnType = mlir::LLVM::LLVMType::getFunctionTy(llvmI32Ty, llvmI32Ty,
                                                          /*isVarArg=*/false);

    mlir::OpBuilder::InsertionGuard insertGuard(*_builder);
    this->_builder->setInsertionPointToStart(_module->getBody());
    return this->_builder->create<mlir::LLVM::LLVMFuncOp>(
        _module->getLoc(), "putchar", llvmFnType);
  }

private:
  std::unique_ptr<mlir::MLIRContext> _context;
  std::unique_ptr<mlir::PassManager> _pm;
  std::unique_ptr<mlir::metal::ModuleOp> _metalModule;
  std::unique_ptr<mlir::ModuleOp> _module;
  std::unique_ptr<mlir::OpBuilder> _builder;
  std::unique_ptr<mlir::Location> _loc;

  void genObjectFile(std::unique_ptr<llvm::Module> &llvmModule,
                     llvm::StringRef outputFilePath) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    auto targetTriple = llvm::sys::getDefaultTargetTriple();

    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
    if (!target) {
      llvm::errs() << error;
      exit(EXIT_FAILURE);
    }

    auto cpu = "generic";
    auto features = "";
    llvm::TargetOptions opt;
    auto rm = llvm::Optional<llvm::Reloc::Model>();
    auto targetMachine =
        target->createTargetMachine(targetTriple, cpu, features, opt, rm);

    llvmModule->setTargetTriple(targetTriple);
    llvmModule->setDataLayout(targetMachine->createDataLayout());

    std::error_code ec;
    llvm::raw_fd_ostream dest(outputFilePath, ec, llvm::sys::fs::F_None);
    if (ec) {
      llvm::errs() << "Could not open file: " << ec.message();
      exit(EXIT_FAILURE);
    }

    llvm::legacy::PassManager pass;
    auto fileType = llvm::CGFT_ObjectFile;
    if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, fileType))
      exit(EXIT_FAILURE);

    pass.run(*llvmModule);

    dest.flush();
  }
};

#endif // METAL_DRIVER_H
