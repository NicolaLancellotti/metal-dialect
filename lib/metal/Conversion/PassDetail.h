//===- PassDetail.h - Conversion Pass class details -------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_PASSDETAIL_H
#define METAL_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {

// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace LLVM {
class LLVMDialect;
} // namespace LLVM

#define GEN_PASS_CLASSES
#include "metal/Conversion/Passes.h.inc"

} // namespace mlir

#endif // METAL_PASSDETAIL_H
