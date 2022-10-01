//===--- ConvertToMetalShadingLanguage.cpp -----------------------------------//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal/IR/MetalDialect.h"
#include "metal/Target/MetalShadingLanguage.h"
#include "metal/Target/ModuleTranslation.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir::metal;

mlir::LogicalResult
mlir::metal::translateModuleToMetalShadingLanguage(mlir::ModuleOp m,
                                                   raw_ostream &output) {
  return ModuleTranslation::translateModule(m, output);
}

namespace mlir {
void registerToMSLTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-msl",
      [](ModuleOp module, raw_ostream &output) {
        return ModuleTranslation::translateModule(module, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<arith::ArithmeticDialect, func::FuncDialect,
                        LLVM::LLVMDialect, metal::MetalDialect,
                        mlir::memref::MemRefDialect>();
      });
}

} // namespace mlir
