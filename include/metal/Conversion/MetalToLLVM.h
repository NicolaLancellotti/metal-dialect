//===--- MetalToLLVM.h ------------------------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_METALTOLLVM_H
#define METAL_METALTOLLVM_H

#include <memory>

namespace mlir {

class MLIRContext;
class RewritePatternSet;
class Pass;

namespace metal {
void populateMetalToLLVMConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx);
} // end namespace metal
} // end namespace mlir

#endif // METAL_METALTOLLVM_H
