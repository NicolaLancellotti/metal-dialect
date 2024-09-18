//===--- ModuleTranslation.cpp -----------------------------------------------//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal/Target/ModuleTranslation.h"
#include "metal/IR/MetalOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir::metal;

struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

#define INDENT()                                                               \
  Indent level_(_curIndent);                                                   \
  indent();

static llvm::StringRef typeToString(mlir::Type type) {
  if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(type))
    switch (intTy.getWidth()) {
    case 1:
      return "bool";
    case 8:
      return intTy.isSigned() ? "int8_t" : "uint8_t";
    case 16:
      return intTy.isSigned() ? "int16_t" : "uint16_t";
    case 32:
      return intTy.isSigned() ? "int32_t" : "uint32_t";
    case 64:
      return intTy.isSigned() ? "int64_t" : "uint64_t";
    default:
      llvm_unreachable("wrong type");
    }
  if (type.isF16())
    return "half";
  if (type.isF32())
    return "float";
  llvm_unreachable("wrong type");
}

void ModuleTranslation::indent() {
  for (int i = 0; i < _curIndent; i++)
    _output << "  ";
}

llvm::LogicalResult ModuleTranslation::translateModule(mlir::ModuleOp m,
                                                       raw_ostream &output) {
  for (auto module : m.getOps<mlir::metal::ModuleOp>()) {
    ModuleTranslation translator{module, output};
    translator.translateKernels();
    output.flush();
  }
  return mlir::success();
}

void ModuleTranslation::translateVarName(mlir::Value memref) {
  auto opInst = memref.getDefiningOp();
  if (opInst == nullptr) {
    _output << "v" << _buffers[memref.getAsOpaquePointer()];
  } else if (auto op = dyn_cast<mlir::metal::AllocaOp>(opInst)) {
    _output << "v" << _alloca[opInst];
  } else {
    llvm_unreachable("llvm_unreachable");
  }
}

void ModuleTranslation::translateKernels() {
  for (auto &op : _metalModule.getOps()) {
    if (auto kernelOp = dyn_cast<mlir::metal::KernelOp>(op)) {
      _varCount = 0;
      _buffers = {};
      translateKernel(kernelOp);
      _output << "\n\n";
    } else if (isa<mlir::metal::ConstantOp, mlir::metal::ModuleEndOp>(op)) {
      // do nothing
    } else {
      llvm_unreachable("unexpected operation");
    }
  }
}

void ModuleTranslation::translateKernel(mlir::metal::KernelOp op) {
  _output << "kernel void " << op.getName() << "(\n";
  for (auto tuple : llvm::zip(op.getBuffers(), op.getAddressSpaceDevice())) {
    auto buffer = std::get<0>(tuple);
    auto memRef = llvm::cast<mlir::metal::MetalMemRefType>(buffer.getType());
    auto stringType = typeToString(memRef.getType());

    auto isDevice = llvm::cast<BoolAttr>(std::get<1>(tuple)).getValue();
    _output << (isDevice ? "  device " : "  constant ");
    _output << stringType << " *v" << _varCount << " [[buffer(" << _varCount
            << ")]],\n";
    _varCount++;
  }
  _output << "  uint3 id [[thread_position_in_grid]])\n";

  auto firstBlock = op.getBodyRegion().getBlocks().begin();
  for (auto const &it : llvm::enumerate(firstBlock->getArguments()))
    _buffers[it.value().getAsOpaquePointer()] = it.index();

  translate(op.getBodyRegion());
}

void ModuleTranslation::printDelim() {
  if (inWhileCondition) {
    _output << ",";
  } else {
    _output << ";";
  }
}

bool ModuleTranslation::isStatementPrintable(Operation *opInst) {
  auto printable = false;
  llvm::TypeSwitch<Operation *>(opInst)
      .Case<mlir::metal::AllocaOp, mlir::metal::StoreOp, mlir::metal::IfOp,
            mlir::metal::WhileOp, mlir::metal::ReturnOp>(
          [&](auto &op) { printable = true; })
      .Case<mlir::metal::YieldWhileOp, mlir::metal::YieldOp>([&](auto &op) {
        // do nothing
        printable = false;
      })
      .Default([&](Operation *) {
        if (opInst->use_empty()) {
          printable = true;
        }
      });
  return printable;
}

void ModuleTranslation::translateStatement(Operation *opInst) {
  llvm::TypeSwitch<Operation *>(opInst)
      .Case<mlir::metal::AllocaOp, mlir::metal::StoreOp, mlir::metal::IfOp,
            mlir::metal::WhileOp, mlir::metal::ReturnOp>(
          [&](auto &op) { translate(op); })
      .Case<mlir::metal::YieldWhileOp, mlir::metal::YieldOp>([&](auto &op) {
        // do nothing;
      })
      .Default([&](Operation *) {
        if (opInst->use_empty()) {
          translateValue(opInst);
          printDelim();
        }
      });
}

void ModuleTranslation::translate(mlir::metal::AllocaOp op) {
  auto memRef = llvm::cast<MetalMemRefType>(op.getResult().getType());
  auto stringType = typeToString(memRef.getType());
  _output << stringType << " v" << _varCount << "[" << memRef.getSize() << "]";
  _alloca[op] = _varCount++;
  _output << ";";
}

void ModuleTranslation::translate(mlir::metal::StoreOp op) {
  translateVarName(op.getMemref());
  _output << "[";
  translateValue(op.getIndex().getDefiningOp());
  _output << "] = ";
  translateValue(op.getValue().getDefiningOp());
  printDelim();
}

void ModuleTranslation::translate(IfOp op) {
  _output << "if (";
  translateValue(op.getCondition().getDefiningOp());
  _output << ") ";

  translate(op.getThenRegion());

  auto &elseRegion = op.getElseRegion();
  if (elseRegion.getBlocks().size()) {
    _output << " else ";
    translate(elseRegion);
  }
}

void ModuleTranslation::translate(WhileOp op) {
  _output << "while (";

  auto &conditionRegion = op.getConditionRegion();
  {
    inWhileCondition = true;
    for (auto &op : conditionRegion.getOps()) {
      translateStatement(&op);
      if (isStatementPrintable(&op))
        _output << " ";
    }
    inWhileCondition = false;
  }
  auto conditionOp =
      dyn_cast<YieldWhileOp>(conditionRegion.back().getTerminator());
  translateValue(conditionOp);
  _output << ") ";

  auto &bodyRegion = op.getBodyRegion();
  translate(bodyRegion);
}

void ModuleTranslation::translate(ReturnOp op) { _output << "return;"; }

void ModuleTranslation::translate(mlir::Region &region) {
  _output << "{";
  {
    INDENT();
    for (auto &op : region.getOps()) {
      if (isStatementPrintable(&op)) {
        _output << "\n";
        indent();
      }
      translateStatement(&op);
    }
  }
  _output << "\n";
  indent();
  _output << "}";
}

void ModuleTranslation::translateValue(Operation *opInst) {
  llvm::TypeSwitch<Operation *>(opInst)
      .Case<mlir::metal::ConstantOp, mlir::metal::GetElementOp,
            mlir::metal::ThreadIdOp, mlir::metal::CastOp,
            mlir::metal::UnaryExpOp, mlir::metal::BinaryExpOp,
            mlir::metal::YieldWhileOp>([&](auto &op) { translate(op); })
      .Default([&](Operation *) { llvm_unreachable("Unexpected operation"); });
}

void ModuleTranslation::translate(mlir::metal::ConstantOp op) {
  if (auto v = llvm::dyn_cast<BoolAttr>(op.getValue()))
    _output << (v.getValue() ? "true" : "false");
  else if (auto v = llvm::dyn_cast<IntegerAttr>(op.getValue()))
    _output << v.getValue();
  else if (auto v = llvm::dyn_cast<FloatAttr>(op.getValue()))
    _output << v.getValueAsDouble();
  else
    llvm_unreachable("Unexpected constant");
}

void ModuleTranslation::translate(mlir::metal::GetElementOp op) {
  translateVarName(op.getMemref());
  _output << "[";
  translateValue(op.getIndex().getDefiningOp());
  _output << "]";
}

void ModuleTranslation::translate(mlir::metal::ThreadIdOp op) {
  _output << "id." << op.getDimension();
}

void ModuleTranslation::translate(mlir::metal::CastOp op) {
  _output << typeToString(op.getType());
  _output << "(";
  translateValue(op.getArgument().getDefiningOp());
  _output << ")";
}

void ModuleTranslation::translate(mlir::metal::UnaryExpOp op) {
  _output << "(";
  using OP = mlir::metal::UnaryExpOperator;
  switch (op.getUnaryOperator()) {
  case OP::minusOp:
    _output << "-";
    break;
  case OP::notOp:
    _output << "!";
    break;
  }
  translateValue(op.getArgument().getDefiningOp());
  _output << ") ";
}

void ModuleTranslation::translate(mlir::metal::BinaryExpOp op) {
  _output << "(";
  translateValue(op.getLhs().getDefiningOp());
  _output << ") ";

  using OP = mlir::metal::BinaryExpOperator;
  switch (op.getBinaryOperator()) {
  case OP::addOp:
    _output << "+";
    break;
  case OP::subOp:
    _output << "-";
    break;
  case OP::mulOp:
    _output << "*";
    break;
  case OP::divOp:
    _output << "/";
    break;
  case OP::remOp:
    _output << "%";
    break;
  case OP::eqOp:
    _output << "==";
    break;
  case OP::neOp:
    _output << "!=";
    break;
  case OP::ltOp:
    _output << "<";
    break;
  case OP::leOp:
    _output << "<=";
    break;
  case OP::gtOp:
    _output << ">";
    break;
  case OP::geOp:
    _output << ">=";
    break;
  case OP::andOp:
    _output << "&&";
    break;
  case OP::orOp:
    _output << "||";
    break;
  }

  _output << " (";
  translateValue(op.getRhs().getDefiningOp());
  _output << ")";
}

void ModuleTranslation::translate(mlir::metal::YieldWhileOp op) {
  translateValue(op.getCondition().getDefiningOp());
}
