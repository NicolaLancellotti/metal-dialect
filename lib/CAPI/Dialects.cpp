//===- Dialects.cpp - CAPI for dialects -----------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal-c/Dialects.h"
#include "metal/IR/MetalDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Metal, metal, mlir::metal::MetalDialect)
