//===--- MetalShadingLanguage.h ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef METAL_METALSHADINGLANGUAGE_H
#define METAL_METALSHADINGLANGUAGE_H

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class ModuleOp;

namespace metal {
mlir::LogicalResult translateModuleToMetalShadingLanguage(mlir::ModuleOp m,
                                                          raw_ostream &output);

} // end namespace metal
} // end namespace mlir

#endif // METAL_METALSHADINGLANGUAGE_H
