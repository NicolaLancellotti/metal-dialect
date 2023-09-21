//===- MetalTypes.td - Metal dialect types ---------------*- tablegen -*-===//
//
// This source file is part of the Metal open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_TYPES
#define METAL_TYPES

include "mlir/IR/AttrTypeBase.td"
include "metal/IR/MetalDialect.td"

//===----------------------------------------------------------------------===//
// Metal type definitions
//===----------------------------------------------------------------------===//

class Metal_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Metal_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Metal_MemRefType : Metal_Type<"MetalMemRef", "memref"> {
    let summary = "Metal memref type";
    let parameters = (ins "Type":$type, "unsigned":$size);
    let hasCustomAssemblyFormat = 1;
}

#endif // METAL_TYPES
