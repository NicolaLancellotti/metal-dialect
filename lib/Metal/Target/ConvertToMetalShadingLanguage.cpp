//===--- ConvertToMetalShadingLanguage.cpp -----------------------------------//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "Metal/Dialect/MetalOps.h"
#include "Metal/Target/MetalShadingLanguage.h"
#include "Metal/Target/ModuleTranslation.h"
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
      "mlir-to-msl", [](ModuleOp module, raw_ostream &output) {
        return ModuleTranslation::translateModule(module, output);
      });
}
} // namespace mlir