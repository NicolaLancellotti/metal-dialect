//===--- MetalDialect.cpp - Metal dialect ---------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Metal/Dialect/MetalDialect.h"
#include "Metal/Dialect/MetalOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::metal;

void MetalDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Metal/Dialect/MetalOps.cpp.inc"
      >();
  addTypes<MetalMemRefType>();
}

mlir::Operation *MetalDialect::materializeConstant(mlir::OpBuilder &builder,
                                                   mlir::Attribute value,
                                                   mlir::Type type,
                                                   mlir::Location loc) {
  return builder.create<mlir::metal::ConstantOp>(loc, value);
}

mlir::Type MetalDialect::parseType(mlir::DialectAsmParser &parser) const {
  Type type;
  if (parser.parseKeyword("memref") || parser.parseLess())
    return Type();

  if (mlir::succeeded(parser.parseOptionalQuestion())) {
    if (parser.parseKeyword("x") || parser.parseType(type) ||
        parser.parseGreater())
      return Type();
    return MetalMemRefType::get(getContext(), type, 0);
  }

  uint32_t size;
  if (parser.parseInteger(size) || parser.parseKeyword("x") ||
      parser.parseType(type) || parser.parseGreater())
    return Type();
  return MetalMemRefType::get(getContext(), type, size);
}

void MetalDialect::printType(mlir::Type type,
                             mlir::DialectAsmPrinter &printer) const {
  MetalMemRefType memRef = type.cast<MetalMemRefType>();
  auto size = memRef.getSize();
  printer << "memref<";
  if (size > 0)
    printer << size;
  else
    printer << "?";
  printer << " x " << memRef.getType() << ">";
}