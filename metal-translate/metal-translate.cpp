//===--- metal-translate.cpp ------------------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#include "metal/IR/MetalDialect.h"

namespace mlir {
void registerToMSLTranslation();
} // end namespace mlir

int main(int argc, char **argv) {
  mlir::registerAllTranslations();
  mlir::registerToMSLTranslation();

  return failed(
      mlir::mlirTranslateMain(argc, argv, "Metal Translation Testing Tool"));
}
