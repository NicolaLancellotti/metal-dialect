//===--- MetalOps.td - Metal dialect ops -------------------*- tablegen -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_OPS
#define METAL_OPS

include "MetalDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Metal_MemRefType :
Type<CPred<"$_self.isa<MetalMemRefType>()">, "Metal MemRefType">;

def Metal_Type : AnyTypeOf<[I1, UI8, UI16, UI32, UI64, SI8, SI16, SI32, SI64,
                            F16, F32]>;

//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

def ModuleOp : Metal_Op<"module", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"ModuleEndOp">]> {

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<"OpBuilder &builder, OperationState &result">];

  let extraClassDeclaration = [{
    void insert(Block::iterator insertPt, Operation *op);
    void push_back(Operation *op);
  }];

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return parseModuleOp(parser, result); }];
}

def ModuleEndOp : Metal_Op<"module_end", [
    Terminator, HasParent<"ModuleOp">]> {
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//

def KernelOp : Metal_Op<"kernel", [SingleBlockImplicitTerminator<"ReturnOp">,
    HasParent<"ModuleOp">]> {
  let arguments = (ins StrAttr:$name, ArrayAttr:$address_space_device);

  let regions = (region SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<"OpBuilder &b, OperationState &state, StringRef name, "
              "llvm::SmallVectorImpl<Type> &buffers, "
              "llvm::SmallVectorImpl<bool> &isAddressSpaceDevice">
  ];

  let verifier = [{ return ::verify(*this); }];

  let extraClassDeclaration = [{
    mlir::Block& getEntryBlock();
    mlir::Value getBuffer(uint32_t index);
    mlir::MutableArrayRef<mlir::BlockArgument> getBuffers();
  }];

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return parseKernelOp(parser, result); }];
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : Metal_Op<"return", [NoSideEffect, Terminator]> {
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def ConstantOp : Metal_Op<"constant", [ConstantLike, NoSideEffect]> {
  let arguments = (ins AnyAttr:$value);
  let results = (outs Metal_Type);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, Attribute value">
  ];

  let verifier = [{ return ::verify(*this); }];

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parseConstantOp(parser, result); }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

def AllocaOp : Metal_Op<"alloca"> {
  let results = (outs Metal_MemRefType);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, "
              "MetalMemRefType type", [{
      state.addTypes(type);
    }]>
  ];

  let verifier = [{ return ::verify(*this); }];

  let assemblyFormat = "attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def StoreOp : Metal_Op<"store"> {
  let arguments = (ins Metal_Type:$value, Metal_MemRefType:$memref, UI32:$index);

  let verifier = [{ return ::verify(*this); }];

  let assemblyFormat = "$value `,` $memref `[` $index `]`  attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// GetElementOp
//===----------------------------------------------------------------------===//

def GetElementOp : Metal_Op<"get_element", [NoSideEffect]> {
  let arguments = (ins Metal_MemRefType:$memref, UI32:$index);
  let results = (outs Metal_Type:$result);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, Value memref, "
              "Value index">];

  let verifier = [{ return ::verify(*this); }];

  let assemblyFormat = "$memref `[` $index `]`  attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// ThreadIdOp
//===----------------------------------------------------------------------===//

def ThreadIdOp : Metal_Op<"thread_id", [NoSideEffect]> {
  let arguments = (ins StrAttr:$dimension);
  let results = (outs UI32);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, StringRef dimension">];

  let verifier = [{ return ::verify(*this); }];

  let assemblyFormat = "$dimension attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def CastOp : Metal_Op<"cast", [NoSideEffect]> {
  let arguments = (ins Metal_Type:$argument);
  let results = (outs Metal_Type);

  let assemblyFormat = "$argument  attr-dict `:` functional-type(operands, results)";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// UnaryExpOp
//===----------------------------------------------------------------------===//

def UnaryExpOperatorNot : I64EnumAttrCase<"notOp", 0>;
def UnaryExpOperatorMinus : I64EnumAttrCase<"minusOp", 1>;

def UnaryExpOperator : I64EnumAttr<
    "UnaryExpOperator", "",
    [UnaryExpOperatorNot, UnaryExpOperatorMinus]> {
  let cppNamespace = "mlir::metal";
}

def UnaryExpOp : Metal_Op<"unary_exp", [NoSideEffect]> {
  let arguments = (ins UnaryExpOperator:$unaryOperator, Metal_Type:$argument);
  let results = (outs Metal_Type);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, UnaryExpOperator "
              "unaryOperator, Value argument">
  ];

  let verifier = [{ return ::verify(*this); }];

  let assemblyFormat = "$argument `,` $unaryOperator attr-dict `:` functional-type(operands, results)";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// BinaryExpOp
//===----------------------------------------------------------------------===//

def BinaryExpOperatorAdd : I64EnumAttrCase<"addOp", 0>;
def BinaryExpOperatorSub : I64EnumAttrCase<"subOp", 1>;
def BinaryExpOperatorMul : I64EnumAttrCase<"mulOp", 2>;
def BinaryExpOperatorDiv : I64EnumAttrCase<"divOp", 3>;
def BinaryExpOperatorRem : I64EnumAttrCase<"remOp", 4>;
def BinaryExpOperatorEq  : I64EnumAttrCase<"eqOp", 5>;
def BinaryExpOperatorNe  : I64EnumAttrCase<"neOp", 6>;
def BinaryExpOperatorLt : I64EnumAttrCase<"ltOp", 7>;
def BinaryExpOperatorLe : I64EnumAttrCase<"leOp", 8>;
def BinaryExpOperatorGt : I64EnumAttrCase<"gtOp", 9>;
def BinaryExpOperatorGe : I64EnumAttrCase<"geOp", 10>;
def BinaryExpOperatorAnd : I64EnumAttrCase<"andOp", 11>;
def BinaryExpOperatorOr : I64EnumAttrCase<"orOp", 12>;

def BinaryExpOperator : I64EnumAttr<
    "BinaryExpOperator", "",
    [BinaryExpOperatorAdd,
     BinaryExpOperatorSub,
     BinaryExpOperatorMul,
     BinaryExpOperatorDiv,
     BinaryExpOperatorRem,
     BinaryExpOperatorEq,
     BinaryExpOperatorNe,
     BinaryExpOperatorLt,
     BinaryExpOperatorLe,
     BinaryExpOperatorGt,
     BinaryExpOperatorGe,
     BinaryExpOperatorAnd,
     BinaryExpOperatorOr]> {
  let cppNamespace = "mlir::metal";
}

def BinaryExpOp : Metal_Op<"binary_exp", [NoSideEffect]> {
  let arguments = (ins BinaryExpOperator:$binaryOperator, Metal_Type:$lhs, Metal_Type:$rhs);
  let results = (outs Metal_Type);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, "
              "BinaryExpOperator binaryOperator, Value lhs, Value rhs">
  ];

  let verifier = [{ return ::verify(*this); }];

  let assemblyFormat = "$lhs `,` $rhs `,` $binaryOperator  attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def YieldOp : Metal_Op<"yield", [NoSideEffect, Terminator]> {
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

def IfOp : Metal_Op<"if", [RecursiveSideEffects, NoRegionArguments]> {
  let arguments = (ins I1:$condition);

  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, Value cond, "
              "function_ref<void(OpBuilder &, Location)> thenBuilder, "
              "function_ref<void(OpBuilder &, Location)> elseBuilder =  nullptr">
  ];

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parseIfOp(parser, result); }];
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

def WhileOp : Metal_Op<"while", [RecursiveSideEffects, NoRegionArguments]> {
  let regions = (region SizedRegion<1>:$conditionRegion, SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, "
              "function_ref<void(OpBuilder &, Location)> conditionBuilder, "
              "function_ref<void(OpBuilder &, Location)> bodyBuilder">
  ];

  let verifier = [{ return ::verify(*this); }];

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parseWhileOp(parser, result); }];
}

def YieldWhileOp : Metal_Op<"while_yield", [NoSideEffect, Terminator,
                                            HasParent<"WhileOp">]> {
  let arguments = (ins I1:$condition);

  let assemblyFormat = "$condition attr-dict ";
}


//===----------------------------------------------------------------------===//
// Runtime - Device
//===----------------------------------------------------------------------===//

def ReleaseOp : Metal_Op<"release"> {
  let arguments = (ins Index:$ref);

  let assemblyFormat = "$ref attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// Runtime - Device
//===----------------------------------------------------------------------===//

def DeviceMakeDefaultOp : Metal_Op<"device_make_default"> {
  let results = (outs Index:$device);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result">
  ];

  let assemblyFormat = "attr-dict `:` type(results)";
}

def DeviceMakeCommandQueueOp : Metal_Op<"device_make_command_queue"> {
  let arguments = (ins Index:$device);
  let results = (outs Index:$commandQueue);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, Value device">
  ];

  let assemblyFormat = "$device attr-dict `:` functional-type(operands, results)";
}

def DeviceMakeBufferOp : Metal_Op<"device_make_buffer"> {
  let arguments = (ins Index:$device, I1:$isStorageModeManaged,
                       I64:$count, I64:$sizeType);

  let results = (outs Index:$bufferOp);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, Value device, "
              "Value isStorageModeManaged, Value count, Value sizeType">
  ];

  let assemblyFormat = "$device `,`$isStorageModeManaged `,` $count `,` $sizeType "
                       "attr-dict `:` functional-type(operands, results)";
}



//===----------------------------------------------------------------------===//
// Runtime - Buffer
//===----------------------------------------------------------------------===//

def BufferGetContentsOp : Metal_Op<"buffer_get_contents"> {
  let arguments = (ins Index:$device);

  let results = (outs AnyMemRef);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, Value device, "
              "Type elementType">
  ];

  let verifier = [{ return ::verify(*this); }];

  let assemblyFormat = "$device attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Runtime - CommandQueue
//===----------------------------------------------------------------------===//

def CommandQueueMakeCommandBufferOp : Metal_Op<"command_queue_make_command_buffer"> {
  let arguments = (ins Index:$commandQueue, I64:$width, I64:$height, I64:$depth,
                       StrAttr:$functionName);
  let results = (outs Index);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, Value commandQueue, "
              "StringRef functionName, Value width, Value height, Value depth">
  ];

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return parseCommandQueueMakeCommandBufferOp(parser, result); }];
}

//===----------------------------------------------------------------------===//
// Runtime - CommandBuffer
//===----------------------------------------------------------------------===//

def CommandBufferAddBufferOp : Metal_Op<"command_buffer_add_buffer"> {
  let arguments = (ins Index:$commandBuffer, Index:$bufferRef, I64:$index);

  let assemblyFormat = "$commandBuffer `,`$bufferRef `,` $index "
                        "attr-dict `:` functional-type(operands, results)";
}

def CommandBufferCommitOp : Metal_Op<"command_buffer_commit"> {
  let arguments = (ins Index:$commandBuffer);

  let assemblyFormat = "$commandBuffer attr-dict `:` type(operands)";
}

def CommandBufferWaitUntilCompletedOp : Metal_Op<"command_buffer_wait_until_completed"> {
  let arguments = (ins Index:$commandBuffer);

  let assemblyFormat = "$commandBuffer attr-dict `:` type(operands)";
}

#endif // METAL_OPS
