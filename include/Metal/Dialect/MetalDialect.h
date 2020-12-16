//===--- MetalDialect.h - Metal dialect -------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_METALDIALECT_H
#define METAL_METALDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace metal {

#include "Metal/Dialect/MetalOpsDialect.h.inc"

} // end namespace metal
} // end namespace mlir

#endif // METAL_METALDIALECT_H
