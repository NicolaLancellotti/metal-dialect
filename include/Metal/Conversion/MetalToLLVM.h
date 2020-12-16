//===--- MetalToLLVM.h ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef METAL_METALTOLLVM_H
#define METAL_METALTOLLVM_H

#include <memory>

namespace mlir {

class MLIRContext;
class OwningRewritePatternList;
class Pass;

void populateMetalToLLVMConversionPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *ctx);

std::unique_ptr<mlir::Pass> createLowerMetalToLLVMPass();

} // end namespace mlir

#endif // METAL_METALTOLLVM_H
