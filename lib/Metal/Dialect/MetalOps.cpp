//===--- MetalOps.cpp - Metal dialect ops ---------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "Metal/Dialect/MetalOps.h"
#include "Metal/Dialect/MetalDialect.h"
#include "Metal/Dialect/MetalMemRefType.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

#include "Metal/Dialect/MetalOpsEnums.cpp.inc"

using namespace mlir::metal;

//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

void ModuleOp::build(OpBuilder &builder, OperationState &result) {
  ensureTerminator(*result.addRegion(), builder, result.location);
}

void ModuleOp::insert(Block::iterator insertPt, Operation *op) {
  auto *body = getBody();
  if (insertPt == body->end())
    insertPt = Block::iterator(body->getTerminator());
  body->getOperations().insert(insertPt, op);
}

void ModuleOp::push_back(Operation *op) {
  insert(Block::iterator(getBody()->getTerminator()), op);
}

static void print(mlir::OpAsmPrinter &printer, ModuleOp op) {
  printer << "metal.module";
  printer.printRegion(op.getRegion(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

static mlir::ParseResult parseModuleOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  mlir::Region *body = result.addRegion();
  if (parser.parseRegion(*body, llvm::None, llvm::None))
    return mlir::failure();
  ModuleOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//

void KernelOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                     llvm::SmallVectorImpl<Type> &buffers,
                     llvm::SmallVectorImpl<bool> &isAddressSpaceDevice) {
  result.addTypes(llvm::None);
  result.addAttribute("name", builder.getStringAttr(name));
  result.addAttribute("address_space_device",
                      builder.getBoolArrayAttr(isAddressSpaceDevice));
  OpBuilder::InsertionGuard guard(builder);
  Region *bodyRegion = result.addRegion();
  auto block = builder.createBlock(bodyRegion);
  for (auto type : buffers) {
    auto memref = MetalMemRefType::get(builder.getContext(), type, 0);
    block->addArguments(memref);
  }
}

mlir::Block &KernelOp::getEntryBlock() { return getRegion().front(); }

static mlir::LogicalResult verify(KernelOp op) {
  auto index = -1;
  for (auto it : llvm::enumerate(op.getBuffers())) {
    auto memRef = it.value().getType().dyn_cast<mlir::metal::MetalMemRefType>();
    if (!memRef) {
      index = it.index();
      break;
    }

    auto type = memRef.getType();
    if (type.isF16() || type.isF32() || type.isIndex())
      continue;
    if (auto intTy = type.dyn_cast<mlir::IntegerType>()) {
      switch (intTy.getWidth()) {
      case 1:
      case 8:
      case 16:
      case 32:
      case 64:
        continue;
      }
    }

    index = it.index();
    break;
  }
  if (index != -1)
    return op.emitOpError() << "type #" << index << " must be compatible type";
  else
    return mlir::success();
}

mlir::Value KernelOp::getBuffer(uint32_t index) {
  return bodyRegion().getBlocks().begin()->getArgument(index);
}

mlir::MutableArrayRef<mlir::BlockArgument> KernelOp::getBuffers() {
  return bodyRegion().getBlocks().begin()->getArguments();
}

static void print(mlir::OpAsmPrinter &printer, KernelOp op) {
  printer << "metal.kernel " << op.name();
  printer << " address_space_device";
  printer.printAttribute(op.address_space_device());
  printer.printRegion(op.getRegion(),
                      /*printEntryBlockArgs=*/true,
                      /*printBlockTerminators=*/true);
}

static mlir::ParseResult parseKernelOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  llvm::StringRef name;
  mlir::Region *body = result.addRegion();
  mlir::Attribute value;
  if (parser.parseKeyword(&name) ||
      parser.parseKeyword("address_space_device") ||
      parser.parseAttribute(value, "address_space_device", result.attributes) ||
      parser.parseRegion(*body, llvm::None, llvm::None))
    return mlir::failure();

  result.addAttribute("name", parser.getBuilder().getStringAttr(name));
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void ConstantOp::build(OpBuilder &builder, OperationState &state,
                       Attribute value) {
  state.addAttribute("value", value);
  state.addTypes(value.getType());
}

static mlir::LogicalResult verify(ConstantOp op) {
  auto value = op.value();
  auto type = op.getType();

  if (value.getType() != type)
    return op.emitOpError() << "requires attribute's type (" << value.getType()
                            << ") to match op's return type (" << type << ")";
  return mlir::success();
}

static mlir::ParseResult parseConstantOp(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result) {
  mlir::Attribute value;
  if (parser.parseAttribute(value, "value", result.attributes))
    return mlir::failure();

  result.addTypes(value.getType());
  return mlir::success();
}

static void print(mlir::OpAsmPrinter &printer, ConstantOp op) {
  printer << "metal.constant ";
  printer.printOptionalAttrDict(op.getAttrs(), {"value"});
  printer << op.value();
}

mlir::OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  return valueAttr();
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult verify(AllocaOp op) {
  if (op.getResult().getType().dyn_cast<MetalMemRefType>().getSize() == 0)
    return op.emitOpError() << "memRef size cannot be 0";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Check Index
//===----------------------------------------------------------------------===//

static mlir::LogicalResult
checkIndex(mlir::Operation *op, MetalMemRefType memRef, mlir::Value index) {
  if (auto constantOp = index.getDefiningOp<ConstantOp>()) {
    auto attr = constantOp.value().dyn_cast<mlir::IntegerAttr>();
    uint64_t index = attr.getUInt();
    uint32_t size = memRef.getSize();
    if (size > 0 && index >= size)
      return op->emitOpError()
             << "index " << index << " is past the end of the memRef "
             << "(which contains " << size << " elements)";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult verify(StoreOp op) {
  auto memRef = op.memref().getType().dyn_cast<MetalMemRefType>();
  auto valueType = op.value().getType();
  auto memRefType = memRef.getType();
  if (memRefType != valueType)
    return op.emitOpError() << "requires value's type (" << valueType
                            << ") to match memref type (" << memRefType << ")";

  return checkIndex(op, memRef, op.index());
}

//===----------------------------------------------------------------------===//
// GetElementOp
//===----------------------------------------------------------------------===//

void GetElementOp::build(OpBuilder &builder, OperationState &result,
                         Value memref, Value index) {
  result.addOperands(memref);
  result.addOperands(index);
  auto type = memref.getType().cast<MetalMemRefType>().getType();
  result.types.push_back(type);
};

static mlir::LogicalResult verify(GetElementOp op) {
  auto memRef = op.memref().getType().dyn_cast<MetalMemRefType>();
  auto resultType = op.result().getType();
  auto memRefType = memRef.getType();
  if (memRefType != resultType)
    return op.emitOpError() << "requires memref type (" << memRefType
                            << ") to match return type (" << resultType << ")";

  return checkIndex(op, memRef, op.index());
}

//===----------------------------------------------------------------------===//
// ThreadIdOp
//===----------------------------------------------------------------------===//

void ThreadIdOp::build(OpBuilder &builder, OperationState &result,
                       StringRef dimension) {
  result.addAttribute("dimension", builder.getStringAttr(dimension));
  result.addTypes(builder.getIntegerType(32, false));
};

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult verify(ThreadIdOp op) {
  auto dim = op.dimension();
  if (dim != "x" && dim != "y" && dim != "z")
    return op.emitOpError() << "requires dimension to be `x` or `y` or `z`, "
                            << "found `" << dim << "`";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

mlir::OpFoldResult CastOp::fold(ArrayRef<Attribute> operands) {
  return mlir::impl::foldCastOp(*this);
}

//===----------------------------------------------------------------------===//
// UnaryExpOp
//===----------------------------------------------------------------------===//

void UnaryExpOp::build(OpBuilder &builder, OperationState &result,
                       UnaryExpOperator unaryOperator, Value argument) {
  result.addTypes(argument.getType());
  auto attr = builder.getI64IntegerAttr(static_cast<int64_t>(unaryOperator));
  result.addAttribute("unaryOperator", attr);
  result.addOperands(argument);
}

static mlir::LogicalResult verify(UnaryExpOp op) {
  auto argType = op.getType();
  auto resultType = op.getResult().getType();
  if (argType != resultType)
    return op.emitOpError() << "result type mismatch";

  using OP = mlir::metal::UnaryExpOperator;
  switch (op.unaryOperator()) {
  case OP::notOp:
    if (!argType.isInteger(1))
      return op.emitOpError() << "argument type must be i1";
    break;
  case OP::minusOp:
    if (argType.isInteger(1) ||
        (!argType.isSignedInteger() && !argType.isF16() && !argType.isF32()))
      return op.emitOpError()
             << "argument type must be signed integer or float";
    break;
  }
  return mlir::success();
}

mlir::OpFoldResult UnaryExpOp::fold(ArrayRef<Attribute> operands) {
  auto constant = dyn_cast<mlir::metal::ConstantOp>(argument().getDefiningOp());
  if (!constant)
    return nullptr;

  mlir::OpBuilder builder{constant.getContext()};

  switch (unaryOperator()) {
  case mlir::metal::UnaryExpOperator::notOp: {
    auto value = constant.valueAttr().dyn_cast<mlir::BoolAttr>().getValue();
    return builder.getBoolAttr(!value);
  }
  case mlir::metal::UnaryExpOperator::minusOp: {
    auto attr = constant.valueAttr();
    if (auto intAttr = attr.dyn_cast<mlir::IntegerAttr>()) {
      return builder.getIntegerAttr(attr.getType(), -intAttr.getSInt());
    }
    if (auto floatAttr = attr.dyn_cast<mlir::FloatAttr>()) {
      return builder.getFloatAttr(attr.getType(),
                                  -floatAttr.getValueAsDouble());
    }
    return nullptr;
  }
  }
}

//===----------------------------------------------------------------------===//
// BinaryExpOp
//===----------------------------------------------------------------------===//

void BinaryExpOp::build(OpBuilder &builder, OperationState &result,
                        BinaryExpOperator binaryOperator, Value lhs,
                        Value rhs) {
  using OP = mlir::metal::BinaryExpOperator;
  switch (binaryOperator) {
  case OP::addOp:
  case OP::subOp:
  case OP::mulOp:
  case OP::divOp:
  case OP::remOp:
    result.addTypes(lhs.getType());
    break;
  case OP::eqOp:
  case OP::neOp:
  case OP::ltOp:
  case OP::leOp:
  case OP::gtOp:
  case OP::geOp:
  case OP::andOp:
  case OP::orOp:
    result.addTypes(builder.getI1Type());
    break;
  }

  auto attr = builder.getI64IntegerAttr(static_cast<int64_t>(binaryOperator));
  result.addAttribute("binaryOperator", attr);
  result.addOperands(lhs);
  result.addOperands(rhs);
}

static mlir::LogicalResult verify(BinaryExpOp op) {
  auto lhsType = op.lhs().getType();
  auto rhsType = op.rhs().getType();
  auto resultType = op.getResult().getType();
  if (lhsType != rhsType)
    return op.emitOpError() << "arguments type mismatch";

  using OP = mlir::metal::BinaryExpOperator;
  switch (op.binaryOperator()) {
  case OP::addOp:
  case OP::subOp:
  case OP::mulOp:
  case OP::divOp:
  case OP::remOp:
    if (lhsType != resultType)
      return op.emitOpError() << "result type mismatch";
    break;
  case OP::eqOp:
  case OP::neOp:
  case OP::ltOp:
  case OP::leOp:
  case OP::gtOp:
  case OP::geOp:
  case OP::andOp:
  case OP::orOp:
    if (!resultType.isInteger(1))
      return op.emitOpError() << "result type mismatch";
    break;
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

auto IfOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &result, mlir::Value cond,
    function_ref<void(mlir::OpBuilder &, mlir::Location)> thenBuilder,
    function_ref<void(OpBuilder &, mlir::Location)> elseBuilder) -> void {
  result.addTypes(llvm::None);
  result.addOperands(cond);

  OpBuilder::InsertionGuard guard(builder);

  Region *thenRegion = result.addRegion();
  builder.createBlock(thenRegion);
  thenBuilder(builder, result.location);

  Region *elseRegion = result.addRegion();
  if (elseBuilder) {
    builder.createBlock(elseRegion);
    elseBuilder(builder, result.location);
  }
}

static mlir::ParseResult parseIfOp(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  result.regions.reserve(2);
  mlir::Region *thenRegion = result.addRegion();
  mlir::Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  mlir::OpAsmParser::OperandType condition;
  mlir::Type i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(condition) ||
      parser.resolveOperand(condition, i1Type, result.operands))
    return mlir::failure();

  if (parser.parseRegion(*thenRegion, llvm::None, llvm::None))
    return mlir::failure();

  if (!parser.parseOptionalKeyword("else")) {
    if (parser.parseRegion(*elseRegion, llvm::None, llvm::None))
      return mlir::failure();
  }

  return mlir::success();
}

static void print(mlir::OpAsmPrinter &printer, IfOp op) {
  printer << "metal.if " << op.condition();
  printer.printRegion(op.thenRegion(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);

  auto &elseRegion = op.elseRegion();
  if (!elseRegion.empty()) {
    printer << " else";
    printer.printRegion(elseRegion,
                        /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/true);
  }
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

auto WhileOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &result,
    function_ref<void(mlir::OpBuilder &, mlir::Location)> conditionBuilder,
    function_ref<void(mlir::OpBuilder &, mlir::Location)> bodyBuilder) -> void {
  result.addTypes(llvm::None);
  result.addOperands(llvm::None);

  OpBuilder::InsertionGuard guard(builder);

  Region *conditionRegion = result.addRegion();
  builder.createBlock(conditionRegion);
  conditionBuilder(builder, result.location);

  Region *bodyRegion = result.addRegion();
  builder.createBlock(bodyRegion);
  bodyBuilder(builder, result.location);
}

static mlir::LogicalResult verify(WhileOp op) {
  auto &region = op.conditionRegion();

  for (auto it = region.op_begin(); it != region.op_end(); it++) {
    if (!llvm::isa<ConstantOp, StoreOp, GetElementOp, ThreadIdOp, CastOp,
                   UnaryExpOp, BinaryExpOp, YieldWhileOp>(*it))
      return op.emitOpError()
             << it->getName() << " op not allowed in the condition region";
  }

  return mlir::success();
}

static mlir::ParseResult parseWhileOp(mlir::OpAsmParser &parser,
                                      mlir::OperationState &result) {
  result.regions.reserve(2);
  mlir::Region *conditionRegion = result.addRegion();
  mlir::Region *bodyRegion = result.addRegion();
  if (parser.parseKeyword("condition") ||
      parser.parseRegion(*conditionRegion, llvm::None, llvm::None) ||
      parser.parseKeyword("loop") ||
      parser.parseRegion(*bodyRegion, llvm::None, llvm::None))
    return mlir::failure();

  return mlir::success();
}

static void print(mlir::OpAsmPrinter &printer, WhileOp op) {
  printer << "metal.while condition";
  printer.printRegion(op.conditionRegion(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);

  auto &bodyRegion = op.bodyRegion();
  printer << " loop";
  printer.printRegion(bodyRegion,
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// Runtime - Device
//===----------------------------------------------------------------------===//

void DeviceMakeDefaultOp::build(OpBuilder &builder, OperationState &result) {
  result.addTypes(builder.getIndexType());
};

void DeviceMakeCommandQueueOp::build(OpBuilder &builder, OperationState &result,
                                     Value device) {
  result.addOperands(device);
  result.addTypes(builder.getIndexType());
};

void DeviceMakeBufferOp::build(OpBuilder &builder, OperationState &result,
                               Value device, Value isStorageModeManaged,
                               Value count, Value sizeType) {
  result.addOperands(device);
  result.addOperands(isStorageModeManaged);
  result.addOperands(count);
  result.addOperands(sizeType);
  result.addTypes(builder.getIndexType());
};

//===----------------------------------------------------------------------===//
// Runtime - Buffer
//===----------------------------------------------------------------------===//

void BufferGetContentsOp::build(OpBuilder &builder, OperationState &result,
                                Value device, Type elementType) {
  result.addOperands(device);
  auto memRefType = mlir::MemRefType::get({-1}, elementType);
  result.addTypes(memRefType);
};

static mlir::LogicalResult verify(BufferGetContentsOp op) {
  auto elementType =
      op.getResult().getType().cast<mlir::MemRefType>().getElementType();
  if (elementType.isa<mlir::IntegerType>() || elementType.isF16() ||
      elementType.isF32())
    return mlir::success();

  return op.emitOpError() << "the buffer has an incompatible type";
}

//===----------------------------------------------------------------------===//
// Runtime - CommandQueue
//===----------------------------------------------------------------------===//

void CommandQueueMakeCommandBufferOp::build(OpBuilder &builder,
                                            OperationState &result,
                                            Value commandQueue,
                                            StringRef functionName, Value width,
                                            Value height, Value depth) {
  result.addAttribute("functionName", builder.getStringAttr(functionName));
  result.addOperands(commandQueue);
  result.addOperands(width);
  result.addOperands(height);
  result.addOperands(depth);
  result.addTypes(builder.getIndexType());
};

static void print(mlir::OpAsmPrinter &printer,
                  CommandQueueMakeCommandBufferOp op) {
  printer << "metal.command_queue_make_command_buffer " << op.functionName()
          << " ";
  printer << op.commandQueue() << ", ";
  printer << op.width() << ", ";
  printer << op.height() << ", ";
  printer << op.depth();
  printer << ": (" << op.getOperandTypes() << ") -> ";
  printer.printType(op.getResult().getType());
}

static mlir::ParseResult
parseCommandQueueMakeCommandBufferOp(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result) {
  llvm::StringRef functionName;
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> operands;
  llvm::SmallVector<mlir::Type, 4> operandTypes;
  if (parser.parseKeyword(&functionName) ||
      parser.parseOperandList(operands, 4) || parser.parseColon() ||
      parser.parseLParen() || parser.parseTypeList(operandTypes) ||
      parser.parseRParen() || parser.parseArrowTypeList(result.types))
    return mlir::failure();

  result.addAttribute("functionName",
                      parser.getBuilder().getStringAttr(functionName));

  return parser.resolveOperands(operands, operandTypes,
                                parser.getCurrentLocation(), result.operands);
}

//===----------------------------------------------------------------------===//
// TableGen's op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Metal/Dialect/MetalOps.cpp.inc"