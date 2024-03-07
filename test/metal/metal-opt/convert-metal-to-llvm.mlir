// RUN: metal-opt --convert-metal-to-llvm %s 2>&1 | FileCheck %s

module {
  llvm.func @putchar(i32) -> i32
  metal.module {
    metal.kernel life address_space_device [false, false, true] {
    ^bb0(%arg0: !metal.memref<? x si32>, %arg1: !metal.memref<? x si32>, %arg2: !metal.memref<? x si32>):
      %0 = metal.constant 0 : ui32
      %1 = metal.constant 1 : ui32
      %2 = metal.constant 0 : si32
      %3 = metal.constant 1 : si32
      %4 = metal.constant 2 : si32
      %5 = metal.constant 3 : si32
      %6 = metal.alloca : !metal.memref<1 x si32>
      %7 = metal.get_element %6[%0] : (!metal.memref<1 x si32>, ui32) -> si32
      %8 = metal.thread_id "x" : ui32
      %9 = metal.cast %8 : (ui32) -> si32
      metal.store %9, %6[%0] : si32, !metal.memref<1 x si32>, ui32
      %10 = metal.alloca : !metal.memref<1 x si32>
      %11 = metal.get_element %10[%0] : (!metal.memref<1 x si32>, ui32) -> si32
      %12 = metal.thread_id "y" : ui32
      %13 = metal.cast %12 : (ui32) -> si32
      metal.store %13, %10[%0] : si32, !metal.memref<1 x si32>, ui32
      %14 = metal.alloca : !metal.memref<1 x si32>
      %15 = metal.alloca : !metal.memref<1 x si32>
      %16 = metal.get_element %14[%0] : (!metal.memref<1 x si32>, ui32) -> si32
      %17 = metal.get_element %15[%0] : (!metal.memref<1 x si32>, ui32) -> si32
      %18 = metal.get_element %arg1[%0] : (!metal.memref<? x si32>, ui32) -> si32
      metal.store %18, %14[%0] : si32, !metal.memref<1 x si32>, ui32
      %19 = metal.get_element %arg1[%1] : (!metal.memref<? x si32>, ui32) -> si32
      metal.store %19, %15[%0] : si32, !metal.memref<1 x si32>, ui32
      %20 = metal.alloca : !metal.memref<1 x si32>
      %21 = metal.get_element %20[%0] : (!metal.memref<1 x si32>, ui32) -> si32
      metal.store %2, %20[%0] : si32, !metal.memref<1 x si32>, ui32
      %22 = metal.alloca : !metal.memref<1 x si32>
      %23 = metal.get_element %22[%0] : (!metal.memref<1 x si32>, ui32) -> si32
      %24 = metal.binary_exp %7, %3, subOp : (si32, si32) -> si32
      metal.store %24, %22[%0] : si32, !metal.memref<1 x si32>, ui32
      metal.while condition {
        %32 = metal.binary_exp %7, %3, addOp : (si32, si32) -> si32
        %33 = metal.binary_exp %23, %32, leOp : (si32, si32) -> i1
        metal.while_yield %33
      } loop {
        %32 = metal.alloca : !metal.memref<1 x si32>
        %33 = metal.get_element %32[%0] : (!metal.memref<1 x si32>, ui32) -> si32
        %34 = metal.binary_exp %11, %3, subOp : (si32, si32) -> si32
        metal.store %34, %32[%0] : si32, !metal.memref<1 x si32>, ui32
        metal.while condition {
          %36 = metal.binary_exp %11, %3, addOp : (si32, si32) -> si32
          %37 = metal.binary_exp %33, %36, leOp : (si32, si32) -> i1
          metal.while_yield %37
        } loop {
          %36 = metal.binary_exp %23, %2, geOp : (si32, si32) -> i1
          %37 = metal.binary_exp %23, %16, ltOp : (si32, si32) -> i1
          %38 = metal.binary_exp %33, %2, geOp : (si32, si32) -> i1
          %39 = metal.binary_exp %33, %17, ltOp : (si32, si32) -> i1
          %40 = metal.binary_exp %23, %7, neOp : (si32, si32) -> i1
          %41 = metal.binary_exp %33, %11, neOp : (si32, si32) -> i1
          %42 = metal.binary_exp %40, %41, orOp : (i1, i1) -> i1
          %43 = metal.binary_exp %23, %17, mulOp : (si32, si32) -> si32
          %44 = metal.binary_exp %43, %33, addOp : (si32, si32) -> si32
          %45 = metal.cast %44 : (si32) -> ui32
          %46 = metal.get_element %arg0[%45] : (!metal.memref<? x si32>, ui32) -> si32
          %47 = metal.cast %46 : (si32) -> i1
          %48 = metal.binary_exp %36, %37, andOp : (i1, i1) -> i1
          %49 = metal.binary_exp %48, %38, andOp : (i1, i1) -> i1
          %50 = metal.binary_exp %49, %39, andOp : (i1, i1) -> i1
          %51 = metal.binary_exp %50, %42, andOp : (i1, i1) -> i1
          %52 = metal.binary_exp %51, %47, andOp : (i1, i1) -> i1
          metal.if %52 {
            %54 = metal.binary_exp %21, %3, addOp : (si32, si32) -> si32
            metal.store %54, %20[%0] : si32, !metal.memref<1 x si32>, ui32
            metal.yield
          }
          %53 = metal.binary_exp %33, %3, addOp : (si32, si32) -> si32
          metal.store %53, %32[%0] : si32, !metal.memref<1 x si32>, ui32
          metal.yield
        }
        %35 = metal.binary_exp %23, %3, addOp : (si32, si32) -> si32
        metal.store %35, %22[%0] : si32, !metal.memref<1 x si32>, ui32
        metal.yield
      }
      %25 = metal.alloca : !metal.memref<1 x si32>
      %26 = metal.get_element %25[%0] : (!metal.memref<1 x si32>, ui32) -> si32
      %27 = metal.binary_exp %7, %17, mulOp : (si32, si32) -> si32
      %28 = metal.binary_exp %27, %11, addOp : (si32, si32) -> si32
      metal.store %28, %25[%0] : si32, !metal.memref<1 x si32>, ui32
      %29 = metal.cast %26 : (si32) -> ui32
      %30 = metal.get_element %arg0[%29] : (!metal.memref<? x si32>, ui32) -> si32
      %31 = metal.cast %30 : (si32) -> i1
      metal.if %31 {
        %32 = metal.binary_exp %21, %4, eqOp : (si32, si32) -> i1
        %33 = metal.binary_exp %21, %5, eqOp : (si32, si32) -> i1
        %34 = metal.binary_exp %32, %33, orOp : (i1, i1) -> i1
        metal.if %34 {
          metal.store %3, %arg2[%29] : si32, !metal.memref<? x si32>, ui32
          metal.yield
        } else {
          metal.store %2, %arg2[%29] : si32, !metal.memref<? x si32>, ui32
          metal.yield
        }
        metal.yield
      } else {
        %32 = metal.binary_exp %21, %5, eqOp : (si32, si32) -> i1
        metal.if %32 {
          metal.store %3, %arg2[%29] : si32, !metal.memref<? x si32>, ui32
          metal.yield
        } else {
          metal.store %2, %arg2[%29] : si32, !metal.memref<? x si32>, ui32
          metal.yield
        }
        metal.yield
      }
      metal.return
    }
    metal.module_end
  }
  func.func @main() -> i32 {
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c4_i64 = arith.constant 4 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c10_i32 = arith.constant 10 : i32
    %c49_i32 = arith.constant 49 : i32
    %c48_i32 = arith.constant 48 : i32
    %c5_i32 = arith.constant 5 : i32
    %c5_i32_0 = arith.constant 5 : i32
    %c5_i64 = arith.constant 5 : i64
    %c5_i64_1 = arith.constant 5 : i64
    %c25_i64 = arith.constant 25 : i64
    %0 = metal.device_make_default : index
    %1 = metal.device_make_command_queue %0 : (index) -> index
    %false = arith.constant false
    %2 = metal.device_make_buffer %0, %false, %c25_i64, %c4_i64 : (index, i1, i64, i64) -> index
    %3 = metal.device_make_buffer %0, %false, %c2_i64, %c4_i64 : (index, i1, i64, i64) -> index
    %4 = metal.device_make_buffer %0, %false, %c25_i64, %c4_i64 : (index, i1, i64, i64) -> index
    %5 = metal.command_queue_make_command_buffer life %1, %c5_i64, %c5_i64_1, %c1_i64: (index, i64, i64, i64) -> index
    %6 = metal.buffer_get_contents %2 : (index) -> memref<?xi32>
    memref.store %c1_i32, %6[%c0] : memref<?xi32>
    memref.store %c1_i32, %6[%c1] : memref<?xi32>
    memref.store %c1_i32, %6[%c2] : memref<?xi32>
    memref.store %c1_i32, %6[%c3] : memref<?xi32>
    %7 = metal.buffer_get_contents %3 : (index) -> memref<?xi32>
    memref.store %c5_i32, %7[%c0] : memref<?xi32>
    memref.store %c5_i32_0, %7[%c1] : memref<?xi32>
    %8 = metal.buffer_get_contents %4 : (index) -> memref<?xi32>
    metal.command_buffer_add_buffer %5, %2, %c0_i64 : (index, index, i64) -> ()
    metal.command_buffer_add_buffer %5, %3, %c1_i64 : (index, index, i64) -> ()
    metal.command_buffer_add_buffer %5, %4, %c2_i64 : (index, index, i64) -> ()
    metal.command_buffer_commit %5 : index
    metal.command_buffer_wait_until_completed %5 : index
    metal.release %0 : index
    metal.release %2 : index
    metal.release %3 : index
    metal.release %4 : index
    metal.release %1 : index
    metal.release %5 : index
    return %c0_i32 : i32
  }
}

// CHECK-LABEL:   llvm.func @_MetalRelease(i64)
// CHECK:         llvm.func @_MetalCommandBufferWaitUntilCompleted(i64)
// CHECK:         llvm.func @_MetalCommandBufferCommit(i64)
// CHECK:         llvm.func @_MetalCommandBufferAddBuffer(i64, i64, i64)
// CHECK:         llvm.func @_MetalBufferGetContents(i64, !llvm.ptr)
// CHECK:         llvm.func @_MetalCommandQueueMakeCommandBuffer(i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i64
// CHECK:         llvm.mlir.global internal constant @life("life") {addr_space = 0 : i32}
// CHECK:         llvm.mlir.global internal constant @metallib("./default.metallib\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @_MetalDeviceMakeBuffer(i64, i1, i64, i64) -> i64
// CHECK:         llvm.func @_MetalDeviceMakeCommandQueue(i64) -> i64
// CHECK:         llvm.func @_MetalDeviceMakeDefault() -> i64
// CHECK:         llvm.func @putchar(i32) -> i32

// CHECK-LABEL:   llvm.func @main() -> i32 {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_8:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK:           %[[VAL_10:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.constant(49 : i32) : i32
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(48 : i32) : i32
// CHECK:           %[[VAL_14:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:           %[[VAL_15:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:           %[[VAL_16:.*]] = llvm.mlir.constant(5 : i64) : i64
// CHECK:           %[[VAL_17:.*]] = llvm.mlir.constant(5 : i64) : i64
// CHECK:           %[[VAL_18:.*]] = llvm.mlir.constant(25 : i64) : i64
// CHECK:           %[[VAL_19:.*]] = llvm.call @_MetalDeviceMakeDefault() : () -> i64
// CHECK:           %[[VAL_20:.*]] = llvm.call @_MetalDeviceMakeCommandQueue(%[[VAL_19]]) : (i64) -> i64
// CHECK:           %[[VAL_21:.*]] = llvm.mlir.constant(false) : i1
// CHECK:           %[[VAL_22:.*]] = llvm.call @_MetalDeviceMakeBuffer(%[[VAL_19]], %[[VAL_21]], %[[VAL_18]], %[[VAL_6]]) : (i64, i1, i64, i64) -> i64
// CHECK:           %[[VAL_23:.*]] = llvm.call @_MetalDeviceMakeBuffer(%[[VAL_19]], %[[VAL_21]], %[[VAL_5]], %[[VAL_6]]) : (i64, i1, i64, i64) -> i64
// CHECK:           %[[VAL_24:.*]] = llvm.call @_MetalDeviceMakeBuffer(%[[VAL_19]], %[[VAL_21]], %[[VAL_18]], %[[VAL_6]]) : (i64, i1, i64, i64) -> i64
// CHECK:           %[[VAL_25:.*]] = llvm.mlir.addressof @metallib : !llvm.ptr
// CHECK:           %[[VAL_26:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_27:.*]] = llvm.getelementptr %[[VAL_25]]{{\[}}%[[VAL_26]], %[[VAL_26]]] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<19 x i8>
// CHECK:           %[[VAL_28:.*]] = llvm.mlir.addressof @life : !llvm.ptr
// CHECK:           %[[VAL_29:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_30:.*]] = llvm.getelementptr %[[VAL_28]]{{\[}}%[[VAL_29]], %[[VAL_29]]] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<4 x i8>
// CHECK:           %[[VAL_31:.*]] = llvm.call @_MetalCommandQueueMakeCommandBuffer(%[[VAL_20]], %[[VAL_27]], %[[VAL_30]], %[[VAL_16]], %[[VAL_17]], %[[VAL_4]]) : (i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i64
// CHECK:           %[[VAL_32:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_33:.*]] = llvm.alloca %[[VAL_32]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i32) -> !llvm.ptr
// CHECK:           llvm.call @_MetalBufferGetContents(%[[VAL_22]], %[[VAL_33]]) : (i64, !llvm.ptr) -> ()
// CHECK:           %[[VAL_34:.*]] = llvm.load %[[VAL_33]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_35:.*]] = llvm.extractvalue %[[VAL_34]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_36:.*]] = llvm.getelementptr %[[VAL_35]]{{\[}}%[[VAL_7]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:           llvm.store %[[VAL_2]], %[[VAL_36]] : i32, !llvm.ptr
// CHECK:           %[[VAL_37:.*]] = llvm.extractvalue %[[VAL_34]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_38:.*]] = llvm.getelementptr %[[VAL_37]]{{\[}}%[[VAL_8]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:           llvm.store %[[VAL_2]], %[[VAL_38]] : i32, !llvm.ptr
// CHECK:           %[[VAL_39:.*]] = llvm.extractvalue %[[VAL_34]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_40:.*]] = llvm.getelementptr %[[VAL_39]]{{\[}}%[[VAL_9]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:           llvm.store %[[VAL_2]], %[[VAL_40]] : i32, !llvm.ptr
// CHECK:           %[[VAL_41:.*]] = llvm.extractvalue %[[VAL_34]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_42:.*]] = llvm.getelementptr %[[VAL_41]]{{\[}}%[[VAL_10]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:           llvm.store %[[VAL_2]], %[[VAL_42]] : i32, !llvm.ptr
// CHECK:           %[[VAL_43:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_44:.*]] = llvm.alloca %[[VAL_43]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i32) -> !llvm.ptr
// CHECK:           llvm.call @_MetalBufferGetContents(%[[VAL_23]], %[[VAL_44]]) : (i64, !llvm.ptr) -> ()
// CHECK:           %[[VAL_45:.*]] = llvm.load %[[VAL_44]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_46:.*]] = llvm.extractvalue %[[VAL_45]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_47:.*]] = llvm.getelementptr %[[VAL_46]]{{\[}}%[[VAL_7]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:           llvm.store %[[VAL_14]], %[[VAL_47]] : i32, !llvm.ptr
// CHECK:           %[[VAL_48:.*]] = llvm.extractvalue %[[VAL_45]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_49:.*]] = llvm.getelementptr %[[VAL_48]]{{\[}}%[[VAL_8]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:           llvm.store %[[VAL_15]], %[[VAL_49]] : i32, !llvm.ptr
// CHECK:           %[[VAL_50:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_51:.*]] = llvm.alloca %[[VAL_50]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i32) -> !llvm.ptr
// CHECK:           llvm.call @_MetalBufferGetContents(%[[VAL_24]], %[[VAL_51]]) : (i64, !llvm.ptr) -> ()
// CHECK:           %[[VAL_52:.*]] = llvm.load %[[VAL_51]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           llvm.call @_MetalCommandBufferAddBuffer(%[[VAL_31]], %[[VAL_22]], %[[VAL_3]]) : (i64, i64, i64) -> ()
// CHECK:           llvm.call @_MetalCommandBufferAddBuffer(%[[VAL_31]], %[[VAL_23]], %[[VAL_4]]) : (i64, i64, i64) -> ()
// CHECK:           llvm.call @_MetalCommandBufferAddBuffer(%[[VAL_31]], %[[VAL_24]], %[[VAL_5]]) : (i64, i64, i64) -> ()
// CHECK:           llvm.call @_MetalCommandBufferCommit(%[[VAL_31]]) : (i64) -> ()
// CHECK:           llvm.call @_MetalCommandBufferWaitUntilCompleted(%[[VAL_31]]) : (i64) -> ()
// CHECK:           llvm.call @_MetalRelease(%[[VAL_19]]) : (i64) -> ()
// CHECK:           llvm.call @_MetalRelease(%[[VAL_22]]) : (i64) -> ()
// CHECK:           llvm.call @_MetalRelease(%[[VAL_23]]) : (i64) -> ()
// CHECK:           llvm.call @_MetalRelease(%[[VAL_24]]) : (i64) -> ()
// CHECK:           llvm.call @_MetalRelease(%[[VAL_20]]) : (i64) -> ()
// CHECK:           llvm.call @_MetalRelease(%[[VAL_31]]) : (i64) -> ()
// CHECK:           llvm.return %[[VAL_1]] : i32
// CHECK:         }
