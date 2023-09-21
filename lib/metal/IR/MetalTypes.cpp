//===- MetalTypes.cpp - Metal dialect types ---------------------*- C++ -*-===//
//
// This source file is part of the Metal open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal/IR/MetalTypes.h"
#include "metal/IR/MetalDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::metal;

#define GET_TYPEDEF_CLASSES
#include "metal/IR/MetalOpsTypes.cpp.inc"

void MetalDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "metal/IR/MetalOpsTypes.cpp.inc"
      >();
}

mlir::Type MetalMemRefType::parse(mlir::AsmParser &parser) {
  Type type;
  if (parser.parseLess())
    return Type();

  if (mlir::succeeded(parser.parseOptionalQuestion())) {
    if (parser.parseKeyword("x") || parser.parseType(type) ||
        parser.parseGreater())
      return Type();
    return MetalMemRefType::get(parser.getContext(), type, 0);
  }

  uint32_t size;
  if (parser.parseInteger(size) || parser.parseKeyword("x") ||
      parser.parseType(type) || parser.parseGreater())
    return Type();
  return MetalMemRefType::get(parser.getContext(), type, size);
}

void MetalMemRefType::print(mlir::AsmPrinter &printer) const {
  MetalMemRefType memRef = llvm::cast<MetalMemRefType>(this->getType());
  auto size = memRef.getSize();
  printer << "memref<";
  if (size > 0)
    printer << size;
  else
    printer << "?";
  printer << " x " << memRef.getType() << ">";
}
