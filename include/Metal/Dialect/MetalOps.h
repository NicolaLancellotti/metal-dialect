//===--- MetalOps.h - Metal dialect ops -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef METAL_METALOPS_H
#define METAL_METALOPS_H

#include "Metal/Dialect/MetalMemRefType.h"
#include "Metal/Dialect/MetalOpsEnums.h.inc"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Metal/Dialect/MetalOps.h.inc"

#endif // METAL_METALOPS_H
