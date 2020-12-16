module {
  llvm.func @putchar(!llvm.i64) -> !llvm.i64
  metal.module {
    metal.kernel life address_space_device[false, false, true] {
    ^bb0(%arg0: !metal.memref<? x si32>, %arg1: !metal.memref<? x si32>, %arg2: !metal.memref<? x si32>):  // no predecessors
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
      %24 = metal.binary_exp %7, %3, "subOp" : (si32, si32) -> si32
      metal.store %24, %22[%0] : si32, !metal.memref<1 x si32>, ui32
      metal.while condition {
        %32 = metal.binary_exp %7, %3, "addOp" : (si32, si32) -> si32
        %33 = metal.binary_exp %23, %32, "leOp" : (si32, si32) -> i1
        metal.while_yield %33
      } loop {
        %32 = metal.alloca : !metal.memref<1 x si32>
        %33 = metal.get_element %32[%0] : (!metal.memref<1 x si32>, ui32) -> si32
        %34 = metal.binary_exp %11, %3, "subOp" : (si32, si32) -> si32
        metal.store %34, %32[%0] : si32, !metal.memref<1 x si32>, ui32
        metal.while condition {
          %36 = metal.binary_exp %11, %3, "addOp" : (si32, si32) -> si32
          %37 = metal.binary_exp %33, %36, "leOp" : (si32, si32) -> i1
          metal.while_yield %37
        } loop {
          %36 = metal.binary_exp %23, %2, "geOp" : (si32, si32) -> i1
          %37 = metal.binary_exp %23, %16, "ltOp" : (si32, si32) -> i1
          %38 = metal.binary_exp %33, %2, "geOp" : (si32, si32) -> i1
          %39 = metal.binary_exp %33, %17, "ltOp" : (si32, si32) -> i1
          %40 = metal.binary_exp %23, %7, "neOp" : (si32, si32) -> i1
          %41 = metal.binary_exp %33, %11, "neOp" : (si32, si32) -> i1
          %42 = metal.binary_exp %40, %41, "orOp" : (i1, i1) -> i1
          %43 = metal.binary_exp %23, %17, "mulOp" : (si32, si32) -> si32
          %44 = metal.binary_exp %43, %33, "addOp" : (si32, si32) -> si32
          %45 = metal.cast %44 : (si32) -> ui32
          %46 = metal.get_element %arg0[%45] : (!metal.memref<? x si32>, ui32) -> si32
          %47 = metal.cast %46 : (si32) -> i1
          %48 = metal.binary_exp %36, %37, "andOp" : (i1, i1) -> i1
          %49 = metal.binary_exp %48, %38, "andOp" : (i1, i1) -> i1
          %50 = metal.binary_exp %49, %39, "andOp" : (i1, i1) -> i1
          %51 = metal.binary_exp %50, %42, "andOp" : (i1, i1) -> i1
          %52 = metal.binary_exp %51, %47, "andOp" : (i1, i1) -> i1
          metal.if %52 {
            %54 = metal.binary_exp %21, %3, "addOp" : (si32, si32) -> si32
            metal.store %54, %20[%0] : si32, !metal.memref<1 x si32>, ui32
            metal.yield
          }
          %53 = metal.binary_exp %33, %3, "addOp" : (si32, si32) -> si32
          metal.store %53, %32[%0] : si32, !metal.memref<1 x si32>, ui32
          metal.yield
        }
        %35 = metal.binary_exp %23, %3, "addOp" : (si32, si32) -> si32
        metal.store %35, %22[%0] : si32, !metal.memref<1 x si32>, ui32
        metal.yield
      }
      %25 = metal.alloca : !metal.memref<1 x si32>
      %26 = metal.get_element %25[%0] : (!metal.memref<1 x si32>, ui32) -> si32
      %27 = metal.binary_exp %7, %17, "mulOp" : (si32, si32) -> si32
      %28 = metal.binary_exp %27, %11, "addOp" : (si32, si32) -> si32
      metal.store %28, %25[%0] : si32, !metal.memref<1 x si32>, ui32
      %29 = metal.cast %26 : (si32) -> ui32
      %30 = metal.get_element %arg0[%29] : (!metal.memref<? x si32>, ui32) -> si32
      %31 = metal.cast %30 : (si32) -> i1
      metal.if %31 {
        %32 = metal.binary_exp %21, %4, "eqOp" : (si32, si32) -> i1
        %33 = metal.binary_exp %21, %5, "eqOp" : (si32, si32) -> i1
        %34 = metal.binary_exp %32, %33, "orOp" : (i1, i1) -> i1
        metal.if %34 {
          metal.store %3, %arg2[%29] : si32, !metal.memref<? x si32>, ui32
          metal.yield
        } else {
          metal.store %2, %arg2[%29] : si32, !metal.memref<? x si32>, ui32
          metal.yield
        }
        metal.yield
      } else {
        %32 = metal.binary_exp %21, %5, "eqOp" : (si32, si32) -> i1
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
  func @main() -> i32 {
    %c-1_i32 = constant -1 : i32
    %c0_i32 = constant 0 : i32
    %c1_i32 = constant 1 : i32
    %c0_i64 = constant 0 : i64
    %c1_i64 = constant 1 : i64
    %c2_i64 = constant 2 : i64
    %c4_i64 = constant 4 : i64
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %0 = llvm.mlir.constant(10 : i32) : !llvm.i64
    %1 = llvm.mlir.constant(49 : i32) : !llvm.i64
    %2 = llvm.mlir.constant(48 : i32) : !llvm.i64
    %c100000000_i32 = constant 100000000 : i32
    %c5_i32 = constant 5 : i32
    %c100000000_i64 = constant 100000000 : i64
    %c5_i64 = constant 5 : i64
    %c500000000_i64 = constant 500000000 : i64
    %3 = metal.device_make_default : index
    %4 = metal.device_make_command_queue %3 : (index) -> index
    %false = constant false
    %5 = metal.device_make_buffer %3, %false, %c500000000_i64, %c4_i64 : (index, i1, i64, i64) -> index
    %6 = metal.device_make_buffer %3, %false, %c2_i64, %c4_i64 : (index, i1, i64, i64) -> index
    %7 = metal.device_make_buffer %3, %false, %c500000000_i64, %c4_i64 : (index, i1, i64, i64) -> index
    %8 = metal.command_queue_make_command_buffer life %4, %c100000000_i64, %c5_i64, %c1_i64: (index, i64, i64, i64) -> index
    %9 = metal.buffer_get_contents %5 : (index) -> memref<?xi32>
    store %c1_i32, %9[%c0] : memref<?xi32>
    store %c1_i32, %9[%c1] : memref<?xi32>
    store %c1_i32, %9[%c2] : memref<?xi32>
    store %c1_i32, %9[%c3] : memref<?xi32>
    %10 = metal.buffer_get_contents %6 : (index) -> memref<?xi32>
    store %c100000000_i32, %10[%c0] : memref<?xi32>
    store %c5_i32, %10[%c1] : memref<?xi32>
    %11 = metal.buffer_get_contents %7 : (index) -> memref<?xi32>
    metal.command_buffer_add_buffer %8, %5, %c0_i64 : (index, index, i64) -> ()
    metal.command_buffer_add_buffer %8, %6, %c1_i64 : (index, index, i64) -> ()
    metal.command_buffer_add_buffer %8, %7, %c2_i64 : (index, index, i64) -> ()
    metal.command_buffer_commit %8 : index
    metal.command_buffer_wait_until_completed %8 : index
    metal.release %3 : index
    metal.release %5 : index
    metal.release %6 : index
    metal.release %7 : index
    metal.release %4 : index
    metal.release %8 : index
    return %c0_i32 : i32
  }
}
