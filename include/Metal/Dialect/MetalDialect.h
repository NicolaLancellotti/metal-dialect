//===--- MetalDialect.h - Metal dialect -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
