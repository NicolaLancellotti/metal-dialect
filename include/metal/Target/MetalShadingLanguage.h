//===--- MetalShadingLanguage.h ---------------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_METALSHADINGLANGUAGE_H
#define METAL_METALSHADINGLANGUAGE_H

#include "llvm/Support/LogicalResult.h"

namespace mlir {
class ModuleOp;

namespace metal {
llvm::LogicalResult translateModuleToMetalShadingLanguage(mlir::ModuleOp m,
                                                          raw_ostream &output);

} // end namespace metal
} // end namespace mlir

#endif // METAL_METALSHADINGLANGUAGE_H
