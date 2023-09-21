//===--- MetalDialect.cpp - Metal dialect ---------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal/IR/MetalDialect.h"
#include "metal/IR/MetalOps.h"
#include "metal/IR/MetalTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::metal;

#include "metal/IR/MetalOpsDialect.cpp.inc"

void MetalDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "metal/IR/MetalOps.cpp.inc"
      >();
  registerTypes();
}

mlir::Operation *MetalDialect::materializeConstant(mlir::OpBuilder &builder,
                                                   mlir::Attribute value,
                                                   mlir::Type type,
                                                   mlir::Location loc) {
  return builder.create<mlir::metal::ConstantOp>(loc, cast<TypedAttr>(value));
}
