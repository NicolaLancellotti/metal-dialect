//===--- MetalOps.h - Metal dialect ops -------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_METALOPS_H
#define METAL_METALOPS_H

#include "mlir/IR/BuiltinAttributes.h"

#include "metal/IR/MetalMemRefType.h"
#include "metal/IR/MetalOpsEnums.h.inc"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "metal/IR/MetalOps.h.inc"

#endif // METAL_METALOPS_H
