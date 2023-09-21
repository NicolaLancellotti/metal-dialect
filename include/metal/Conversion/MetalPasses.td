//===--- MetalPasses.td - Metal passes ---------------------*- tablegen -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_PASSES_TD
#define METAL_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertMetalToLLVM : Pass<"convert-metal-to-llvm"> {
  let summary = "Convert Metal operations into the LLVM dialect";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

#endif // METAL_PASSES_TD
