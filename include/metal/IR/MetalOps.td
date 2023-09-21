//===--- MetalOps.td - Metal dialect ops -------------------*- tablegen -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_OPS
#define METAL_OPS

include "metal/IR/MetalTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def Metal_Type : AnyTypeOf<[I1, UI8, UI16, UI32, UI64, SI8, SI16, SI32, SI64,
                            F16, F32]>;

//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

def ModuleOp : Metal_Op<"module", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"ModuleEndOp">]> {
  let summary = "Module operation";

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<(ins)>];

  let hasCustomAssemblyFormat = 1;
}

def ModuleEndOp : Metal_Op<"module_end", [
    Terminator, HasParent<"ModuleOp">]> {
  let summary = "Module end operation";

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//

def KernelOp : Metal_Op<"kernel", [SingleBlockImplicitTerminator<"ReturnOp">,
    HasParent<"ModuleOp">]> {
  let summary = "Kernel operation";

  let arguments = (ins StrAttr:$name, ArrayAttr:$address_space_device);

  let regions = (region SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "llvm::SmallVectorImpl<Type>&":$buffers,
                      "llvm::SmallVectorImpl<bool>&":$isAddressSpaceDevice)>
  ];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    mlir::Block& getEntryBlock();
    mlir::Value getBuffer(uint32_t index);
    mlir::MutableArrayRef<mlir::BlockArgument> getBuffers();
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : Metal_Op<"return", [Pure, Terminator]> {
  let summary = "Return operation";

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def ConstantOp : Metal_Op<"constant", [ConstantLike, Pure, AllTypesMatch<["value", "result"]>,]> {
  let summary = "Constant operation";

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs Metal_Type:$result);

  let builders = [
    OpBuilder<(ins "TypedAttr":$attr)>
  ];

  let hasFolder = 1;

  let assemblyFormat = "$value attr-dict";
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

def AllocaOp : Metal_Op<"alloca"> {
  let summary = "Alloca operation";

  let results = (outs Metal_MemRefType);

  let builders = [
    OpBuilder<(ins "MetalMemRefType":$type), [{
      $_state.addTypes(type);
    }]>
  ];

  let hasVerifier = 1;

  let assemblyFormat = "attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def StoreOp : Metal_Op<"store"> {
  let summary = "Store operation";

  let arguments = (ins Metal_Type:$value, Metal_MemRefType:$memref, UI32:$index);

  let hasVerifier = 1;

  let assemblyFormat = "$value `,` $memref `[` $index `]`  attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// GetElementOp
//===----------------------------------------------------------------------===//

def GetElementOp : Metal_Op<"get_element", [Pure]> {
  let summary = "Get element operation";

  let arguments = (ins Metal_MemRefType:$memref, UI32:$index);
  let results = (outs Metal_Type:$result);

  let builders = [OpBuilder<(ins "Value":$memref, "Value":$index)>];

  let hasVerifier = 1;

  let assemblyFormat = "$memref `[` $index `]`  attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// ThreadIdOp
//===----------------------------------------------------------------------===//

def ThreadIdOp : Metal_Op<"thread_id", [Pure]> {
  let summary = "Thread id operation";

  let arguments = (ins StrAttr:$dimension);
  let results = (outs UI32);

  let builders = [OpBuilder<(ins "StringRef":$dimension)>];

  let hasVerifier = 1;

  let assemblyFormat = "$dimension attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def CastOp : Metal_Op<"cast", [Pure]> {
  let summary = "Cast operation";

  let arguments = (ins Metal_Type:$argument);
  let results = (outs Metal_Type);

  let assemblyFormat = "$argument  attr-dict `:` functional-type(operands, results)";
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

def UnaryExpOp : Metal_Op<"unary_exp", [Pure]> {
  let summary = "Unary expression operation";

  let arguments = (ins UnaryExpOperator:$unaryOperator, Metal_Type:$argument);
  let results = (outs Metal_Type);

  let builders = [
    OpBuilder<(ins "UnaryExpOperator":$unaryOperator, "Value":$argument)>
  ];

  let hasVerifier = 1;

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

def BinaryExpOp : Metal_Op<"binary_exp", [Pure]> {
  let summary = "Binary expression operation";

  let arguments = (ins BinaryExpOperator:$binaryOperator, Metal_Type:$lhs, Metal_Type:$rhs);
  let results = (outs Metal_Type);

  let builders = [
    OpBuilder<(ins "BinaryExpOperator":$binaryOperator, "Value":$lhs, "Value":$rhs)>
  ];

  let hasVerifier = 1;

  let assemblyFormat = "$lhs `,` $rhs `,` $binaryOperator  attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def YieldOp : Metal_Op<"yield", [Pure, Terminator]> {
  let summary = "Yield operation";

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

def IfOp : Metal_Op<"if", [RecursiveMemoryEffects, NoRegionArguments]> {
  let summary = "If-then-else operation";

  let arguments = (ins I1:$condition);

  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let builders = [
    OpBuilder<(ins "Value":$cond, "function_ref<void(OpBuilder &, Location)>":$thenBuilder,
               CArg<"function_ref<void(OpBuilder &, Location)>", "nullptr">:$elseBuilder)>
  ];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

def WhileOp : Metal_Op<"while", [RecursiveMemoryEffects, NoRegionArguments]> {
  let summary = "While operation";

  let regions = (region SizedRegion<1>:$conditionRegion, SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<(ins "function_ref<void(OpBuilder &, Location)>":$conditionBuilder,
                      "function_ref<void(OpBuilder &, Location)>":$bodyBuilder)>
  ];

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;
}

def YieldWhileOp : Metal_Op<"while_yield", [Pure, Terminator,
                                            HasParent<"WhileOp">]> {
  let summary = "Yield while operation";

  let arguments = (ins I1:$condition);

  let assemblyFormat = "$condition attr-dict ";
}


//===----------------------------------------------------------------------===//
// ReleaseOp
//===----------------------------------------------------------------------===//

def ReleaseOp : Metal_Op<"release"> {
  let summary = "Release operation";

  let arguments = (ins Index:$ref);

  let assemblyFormat = "$ref attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// Runtime - Device
//===----------------------------------------------------------------------===//

def DeviceMakeDefaultOp : Metal_Op<"device_make_default"> {
  let summary = "Device make operation";

  let results = (outs Index:$device);

  let builders = [OpBuilder<(ins)>];

  let assemblyFormat = "attr-dict `:` type(results)";
}

def DeviceMakeCommandQueueOp : Metal_Op<"device_make_command_queue"> {
  let summary = "Device make command queue operation";

  let arguments = (ins Index:$device);
  let results = (outs Index:$commandQueue);

  let builders = [
    OpBuilder<(ins "Value":$device)>
  ];

  let assemblyFormat = "$device attr-dict `:` functional-type(operands, results)";
}

def DeviceMakeBufferOp : Metal_Op<"device_make_buffer"> {
  let summary = "Device make buffer operation";

  let arguments = (ins Index:$device, I1:$isStorageModeManaged,
                       I64:$count, I64:$sizeType);

  let results = (outs Index:$bufferOp);

  let builders = [
    OpBuilder<(ins "Value":$device, "Value":$isStorageModeManaged, "Value":$count,
                      "Value":$sizeType)>
  ];

  let assemblyFormat = "$device `,`$isStorageModeManaged `,` $count `,` $sizeType "
                       "attr-dict `:` functional-type(operands, results)";
}



//===----------------------------------------------------------------------===//
// Runtime - Buffer
//===----------------------------------------------------------------------===//

def BufferGetContentsOp : Metal_Op<"buffer_get_contents"> {
  let summary = "Buffer get contents operation";

  let arguments = (ins Index:$device);

  let results = (outs AnyMemRef);

  let builders = [
    OpBuilder<(ins "Value":$device, "Type":$elementType)>
  ];

  let hasVerifier = 1;

  let assemblyFormat = "$device attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Runtime - CommandQueue
//===----------------------------------------------------------------------===//

def CommandQueueMakeCommandBufferOp : Metal_Op<"command_queue_make_command_buffer"> {
  let summary = "Command queue make command buffer operation";

  let arguments = (ins Index:$commandQueue, I64:$width, I64:$height, I64:$depth,
                       StrAttr:$functionName);
  let results = (outs Index);

  let builders = [
    OpBuilder<(ins "Value":$commandQueue, "StringRef":$functionName,
                      "Value":$width, "Value":$height, "Value":$depth)>
  ];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Runtime - CommandBuffer
//===----------------------------------------------------------------------===//

def CommandBufferAddBufferOp : Metal_Op<"command_buffer_add_buffer"> {
  let summary = "Command buffer add buffer operation";

  let arguments = (ins Index:$commandBuffer, Index:$bufferRef, I64:$index);

  let assemblyFormat = "$commandBuffer `,`$bufferRef `,` $index "
                        "attr-dict `:` functional-type(operands, results)";
}

def CommandBufferCommitOp : Metal_Op<"command_buffer_commit"> {
  let summary = "Command buffer commit operation";

  let arguments = (ins Index:$commandBuffer);

  let assemblyFormat = "$commandBuffer attr-dict `:` type(operands)";
}

def CommandBufferWaitUntilCompletedOp : Metal_Op<"command_buffer_wait_until_completed"> {
  let summary = "Command buffer wait until completed operation";

  let arguments = (ins Index:$commandBuffer);

  let assemblyFormat = "$commandBuffer attr-dict `:` type(operands)";
}

#endif // METAL_OPS
