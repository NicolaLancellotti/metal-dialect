//===- Dialects.h - CAPI for dialects -----------------------------*- C -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_C_DIALECTS_H
#define METAL_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Metal, metal);

#ifdef __cplusplus
}
#endif

#endif // METAL_C_DIALECTS_H
