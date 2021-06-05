//===--- ConvertToMetalShadingLanguage.cpp -----------------------------------//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Metal/Dialect/MetalDialect.h"
#include "Metal/Dialect/MetalOps.h"
#include "Metal/Target/MetalShadingLanguage.h"
#include "Metal/Target/ModuleTranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Translation.h"

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
        registry.insert<LLVM::LLVMDialect, metal::MetalDialect,
                        mlir::StandardOpsDialect>();
      });
}

} // namespace mlir