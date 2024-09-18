//===--- ModuleTranslation.h ------------------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_MODULETRANSLATION_H
#define METAL_MODULETRANSLATION_H

#include "metal/IR/MetalOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/FileSystem.h>
#include <map>

namespace mlir {
class Location;
class ModuleOp;
class Operation;

namespace metal {
class KernelOp;

class ModuleTranslation {
public:
  static llvm::LogicalResult translateModule(mlir::ModuleOp m,
                                             raw_ostream &output);

private:
  ModuleTranslation(Operation *module, raw_ostream &output)
      : _metalModule(module), _output(output){};
  mlir::metal::ModuleOp _metalModule;
  std::map<mlir::Operation *, unsigned> _alloca;
  std::map<void *, size_t> _buffers;
  unsigned _varCount = 0;
  bool inWhileCondition = false;
  int _curIndent = 0;
  raw_ostream &_output;

  void indent();
  void printDelim();
  void translateVarName(mlir::Value memref);

  void translateKernels();
  void translateKernel(mlir::metal::KernelOp op);

  bool isStatementPrintable(Operation *opInst);
  void translateStatement(Operation *opInst);
  void translate(mlir::metal::AllocaOp op);
  void translate(mlir::metal::StoreOp op);
  void translate(mlir::metal::IfOp op);
  void translate(mlir::metal::WhileOp op);
  void translate(mlir::metal::ReturnOp op);
  void translate(mlir::Region &region);

  void translateValue(Operation *opInst);
  void translate(mlir::metal::ConstantOp op);
  void translate(mlir::metal::GetElementOp op);
  void translate(mlir::metal::ThreadIdOp op);
  void translate(mlir::metal::CastOp op);
  void translate(mlir::metal::UnaryExpOp op);
  void translate(mlir::metal::BinaryExpOp op);
  void translate(mlir::metal::YieldWhileOp op);
};

} // end namespace metal
} // end namespace mlir

#endif // METAL_MODULETRANSLATION_H
