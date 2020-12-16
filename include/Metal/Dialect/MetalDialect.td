//===--- MetalDialect.td - Metal dialect -------------------*- tablegen -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_DIALECT
#define METAL_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Metal dialect definition.
//===----------------------------------------------------------------------===//

def Metal_Dialect : Dialect {
    let name = "metal";
    let summary = "Metal MLIR dialect.";
    let cppNamespace = "metal";

    let extraClassDeclaration = [{
      Operation *materializeConstant(OpBuilder &builder, Attribute value,
                                     Type type, Location loc) override;

      ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

      void printType(mlir::Type type,
                     mlir::DialectAsmPrinter &printer) const override;
    }];
}

//===----------------------------------------------------------------------===//
// Base metal operation definition.
//===----------------------------------------------------------------------===//

class Metal_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Metal_Dialect, mnemonic, traits>;

#endif // METAL_DIALECT
