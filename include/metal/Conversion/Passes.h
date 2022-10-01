//===--- Passes.h - Metal passes --------------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_PASSES_H
#define METAL_PASSES_H

#include "metal/Conversion/MetalToLLVM.h"

namespace mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "metal/Conversion/Passes.h.inc"

} // namespace mlir

#endif // METAL_PASSES_H
