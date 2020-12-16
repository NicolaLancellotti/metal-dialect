//===--- MetalOps.h - Metal dialect ops -------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_METALOPS_H
#define METAL_METALOPS_H

#include "Metal/Dialect/MetalMemRefType.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Metal/Dialect/MetalOpsEnums.h.inc"

namespace mlir {
namespace metal {

#define GET_OP_CLASSES
#include "Metal/Dialect/MetalOps.h.inc"

} // end namespace metal
} // end namespace mlir

#endif // METAL_METALOPS_H
