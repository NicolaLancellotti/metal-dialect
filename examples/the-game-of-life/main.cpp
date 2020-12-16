#include "utility/Driver.h"
namespace ml = mlir::metal;

#define littleMatrix
#define printMatrix

// -----------------------------------------------------------------------------
// GPU Code

static ml::KernelOp createKernel(mlir::OpBuilder builder, mlir::Location loc) {
  auto si32 = builder.getIntegerType(32, true);
  auto ui32 = builder.getIntegerType(32, false);
  llvm::SmallVector<mlir::Type, 3> buffers = {si32, si32, si32};
  llvm::SmallVector<bool, 3> isAddressSpaceDevice = {false, false, true};
  auto kernel =
      builder.create<ml::KernelOp>(loc, "life", buffers, isAddressSpaceDevice);

  builder.setInsertionPointToStart(&kernel.getEntryBlock());
  auto si32MemRef1 = ml::MetalMemRefType::get(builder.getContext(), si32, 1);

  auto ui32_0 =
      builder.create<ml::ConstantOp>(loc, builder.getUI32IntegerAttr(0));
  auto ui32_1 =
      builder.create<ml::ConstantOp>(loc, builder.getUI32IntegerAttr(1));
  auto si32_0 =
      builder.create<ml::ConstantOp>(loc, builder.getSI32IntegerAttr(0));
  auto si32_1 =
      builder.create<ml::ConstantOp>(loc, builder.getSI32IntegerAttr(1));
  auto si32_2 =
      builder.create<ml::ConstantOp>(loc, builder.getSI32IntegerAttr(2));
  auto si32_3 =
      builder.create<ml::ConstantOp>(loc, builder.getSI32IntegerAttr(3));

  auto buffer0 = kernel.getBuffer(0);
  auto buffer1 = kernel.getBuffer(1);
  auto buffer2 = kernel.getBuffer(2);

  auto xMemRef = builder.create<ml::AllocaOp>(loc, si32MemRef1);
  auto x = builder.create<ml::GetElementOp>(loc, xMemRef, ui32_0);
  auto xsi32 = builder.create<ml::CastOp>(
      loc, si32, builder.create<ml::ThreadIdOp>(loc, "x"));
  builder.create<ml::StoreOp>(loc, xsi32, xMemRef, ui32_0);

  auto yMemRef = builder.create<ml::AllocaOp>(loc, si32MemRef1);
  auto y = builder.create<ml::GetElementOp>(loc, yMemRef, ui32_0);
  auto ysi32 = builder.create<ml::CastOp>(
      loc, si32, builder.create<ml::ThreadIdOp>(loc, "y"));
  builder.create<ml::StoreOp>(loc, ysi32, yMemRef, ui32_0);

  auto rowsMemRef = builder.create<ml::AllocaOp>(loc, si32MemRef1);
  auto columnsMemRef = builder.create<ml::AllocaOp>(loc, si32MemRef1);
  auto rows = builder.create<ml::GetElementOp>(loc, rowsMemRef, ui32_0);
  auto columns = builder.create<ml::GetElementOp>(loc, columnsMemRef, ui32_0);
  {
    builder.create<ml::StoreOp>(
        loc, builder.create<ml::GetElementOp>(loc, buffer1, ui32_0), rowsMemRef,
        ui32_0);
    builder.create<ml::StoreOp>(
        loc, builder.create<ml::GetElementOp>(loc, buffer1, ui32_1),
        columnsMemRef, ui32_0);
  }

  auto liveNeighborsMemRef = builder.create<ml::AllocaOp>(loc, si32MemRef1);
  auto liveNeighbors =
      builder.create<ml::GetElementOp>(loc, liveNeighborsMemRef, ui32_0);
  builder.create<ml::StoreOp>(loc, si32_0, liveNeighborsMemRef, ui32_0);

  auto iMemRef = builder.create<ml::AllocaOp>(loc, si32MemRef1);
  auto i = builder.create<ml::GetElementOp>(loc, iMemRef, ui32_0);
  {
    auto value = builder.create<ml::BinaryExpOp>(
        loc, ml::BinaryExpOperator::subOp, x, si32_1);
    builder.create<ml::StoreOp>(loc, value, iMemRef, ui32_0);
  }

  builder.create<ml::WhileOp>(
      loc,
      [&](mlir::OpBuilder &builder, mlir::Location) {
        auto xPlusOne = builder.create<ml::BinaryExpOp>(
            loc, ml::BinaryExpOperator::addOp, x, si32_1);
        auto boolValue = builder.create<ml::BinaryExpOp>(
            loc, ml::BinaryExpOperator::leOp, i, xPlusOne);
        builder.create<ml::YieldWhileOp>(loc, boolValue);
      },
      [&](mlir::OpBuilder &builder, mlir::Location) {
        auto jMemRef = builder.create<ml::AllocaOp>(loc, si32MemRef1);
        auto j = builder.create<ml::GetElementOp>(loc, jMemRef, ui32_0);
        {
          auto value = builder.create<ml::BinaryExpOp>(
              loc, ml::BinaryExpOperator::subOp, y, si32_1);
          builder.create<ml::StoreOp>(loc, value, jMemRef, ui32_0);
        }

        builder.create<ml::WhileOp>(
            loc,
            [&](mlir::OpBuilder &builder, mlir::Location) {
              auto yPlusOne = builder.create<ml::BinaryExpOp>(
                  loc, ml::BinaryExpOperator::addOp, y, si32_1);
              auto boolValue = builder.create<ml::BinaryExpOp>(
                  loc, ml::BinaryExpOperator::leOp, j, yPlusOne);
              builder.create<ml::YieldWhileOp>(loc, boolValue);
            },
            [&](mlir::OpBuilder &builder, mlir::Location) {
              auto value0 = builder.create<ml::BinaryExpOp>(
                  loc, ml::BinaryExpOperator::geOp, i, si32_0);
              auto value1 = builder.create<ml::BinaryExpOp>(
                  loc, ml::BinaryExpOperator::ltOp, i, rows);
              auto value2 = builder.create<ml::BinaryExpOp>(
                  loc, ml::BinaryExpOperator::geOp, j, si32_0);
              auto value3 = builder.create<ml::BinaryExpOp>(
                  loc, ml::BinaryExpOperator::ltOp, j, columns);

              auto valueB1 = builder.create<ml::BinaryExpOp>(
                  loc, ml::BinaryExpOperator::neOp, i, x);
              auto valueB2 = builder.create<ml::BinaryExpOp>(
                  loc, ml::BinaryExpOperator::neOp, j, y);
              auto value4 = builder.create<ml::BinaryExpOp>(
                  loc, ml::BinaryExpOperator::orOp, valueB1, valueB2);

              auto valueA1 = builder.create<ml::BinaryExpOp>(
                  loc, ml::BinaryExpOperator::mulOp, i, columns);
              auto valueA2 = builder.create<ml::BinaryExpOp>(
                  loc, ml::BinaryExpOperator::addOp, valueA1, j);
              auto valueA2AsUi32 =
                  builder.create<ml::CastOp>(loc, ui32, valueA2);
              auto valueA3 =
                  builder.create<ml::GetElementOp>(loc, buffer0, valueA2AsUi32);
              auto value5 =
                  builder.create<ml::CastOp>(loc, builder.getI1Type(), valueA3);

              auto valueAnd1 = builder.create<ml::BinaryExpOp>(
                  loc, ml::BinaryExpOperator::andOp, value0, value1);
              auto valueAnd2 = builder.create<ml::BinaryExpOp>(
                  loc, ml::BinaryExpOperator::andOp, valueAnd1, value2);
              auto valueAnd3 = builder.create<ml::BinaryExpOp>(
                  loc, ml::BinaryExpOperator::andOp, valueAnd2, value3);
              auto valueAnd4 = builder.create<ml::BinaryExpOp>(
                  loc, ml::BinaryExpOperator::andOp, valueAnd3, value4);
              auto valueAnd5 = builder.create<ml::BinaryExpOp>(
                  loc, ml::BinaryExpOperator::andOp, valueAnd4, value5);

              builder.create<ml::IfOp>(
                  loc, valueAnd5,
                  [&](mlir::OpBuilder &builder, mlir::Location) {
                    auto value = builder.create<ml::BinaryExpOp>(
                        loc, ml::BinaryExpOperator::addOp, liveNeighbors,
                        si32_1);
                    builder.create<ml::StoreOp>(loc, value, liveNeighborsMemRef,
                                                ui32_0);
                    builder.create<ml::YieldOp>(loc);
                  },
                  nullptr);

              auto value = builder.create<ml::BinaryExpOp>(
                  loc, ml::BinaryExpOperator::addOp, j, si32_1);
              builder.create<ml::StoreOp>(loc, value, jMemRef, ui32_0);
              builder.create<ml::YieldOp>(loc);
            });

        auto value = builder.create<ml::BinaryExpOp>(
            loc, ml::BinaryExpOperator::addOp, i, si32_1);
        builder.create<ml::StoreOp>(loc, value, iMemRef, ui32_0);
        builder.create<ml::YieldOp>(loc);
      });

  auto indexMemRef = builder.create<ml::AllocaOp>(loc, si32MemRef1);
  auto index = builder.create<ml::GetElementOp>(loc, indexMemRef, ui32_0);
  {
    auto value = builder.create<ml::BinaryExpOp>(
        loc, ml::BinaryExpOperator::mulOp, x, columns);
    value = builder.create<ml::BinaryExpOp>(loc, ml::BinaryExpOperator::addOp,
                                            value, y);
    builder.create<ml::StoreOp>(loc, value, indexMemRef, ui32_0);
  }

  auto indexAsUi32 = builder.create<ml::CastOp>(loc, ui32, index);
  auto conditionInt =
      builder.create<ml::GetElementOp>(loc, buffer0, indexAsUi32);
  auto conditionBool =
      builder.create<ml::CastOp>(loc, builder.getI1Type(), conditionInt);
  builder.create<ml::IfOp>(
      loc, conditionBool,
      [&](mlir::OpBuilder &builder, mlir::Location) {
        auto is2 = builder.create<ml::BinaryExpOp>(
            loc, ml::BinaryExpOperator::eqOp, liveNeighbors, si32_2);
        auto is3 = builder.create<ml::BinaryExpOp>(
            loc, ml::BinaryExpOperator::eqOp, liveNeighbors, si32_3);
        auto is2or3 = builder.create<ml::BinaryExpOp>(
            loc, ml::BinaryExpOperator::orOp, is2, is3);

        builder.create<ml::IfOp>(
            loc, is2or3,
            [&](mlir::OpBuilder &builder, mlir::Location) {
              builder.create<ml::StoreOp>(loc, si32_1, buffer2, indexAsUi32);
              builder.create<ml::YieldOp>(loc);
            },
            [&](mlir::OpBuilder &builder, mlir::Location) {
              builder.create<ml::StoreOp>(loc, si32_0, buffer2, indexAsUi32);
              builder.create<ml::YieldOp>(loc);
            });

        builder.create<ml::YieldOp>(loc);
      },
      [&](mlir::OpBuilder &builder, mlir::Location) {
        auto condition = builder.create<ml::BinaryExpOp>(
            loc, ml::BinaryExpOperator::eqOp, liveNeighbors, si32_3);

        builder.create<ml::IfOp>(
            loc, condition,
            [&](mlir::OpBuilder &builder, mlir::Location) {
              builder.create<ml::StoreOp>(loc, si32_1, buffer2, indexAsUi32);
              builder.create<ml::YieldOp>(loc);
            },
            [&](mlir::OpBuilder &builder, mlir::Location) {
              builder.create<ml::StoreOp>(loc, si32_0, buffer2, indexAsUi32);
              builder.create<ml::YieldOp>(loc);
            });

        builder.create<ml::YieldOp>(loc);
      });

  builder.create<ml::ReturnOp>(loc);
  return kernel;
}

// -----------------------------------------------------------------------------
// CPU Code

/*
Print matrix:

int32_t i = -1
while (i = i + 1, i < side) {
  int32_t j = -1
  while (j = j - 1,  j < side ) {
    printf("%d", b[i * side + j]);
  }
  printf("\n");
}
printf("\n");

Pseudocode
int32_t i = -1
branch B1

B1:
  i = i + 1
  if i < side branch B2 else branch B3

B2:
  int32_t j = -1
  branch B4

B3:
  printf("\n");
  return

B4:
   j = j + 1
   if j < side branch B5 else branch B6

B5:
  printf("%d", b[i * side + j]);
  branch B4

B6:
  printf("\n");
  branch B1;
 */

static mlir::FuncOp createExecuteKernel(mlir::OpBuilder builder,
                                        mlir::Location loc,
                                        mlir::LLVM::LLVMFuncOp putcharF) {
  llvm::SmallVector<mlir::Type, 0> arg_types;
  llvm::SmallVector<mlir::Type, 1> result_types{builder.getI32Type()};

  // Types
  auto llvmDialect =
      builder.getContext()->getRegisteredDialect<mlir::LLVM::LLVMDialect>();
  auto llvmI32Ty = mlir::LLVM::LLVMType::getInt64Ty(llvmDialect);
  auto i32Ty = builder.getIntegerType(32);
  auto indexTy = builder.getIndexType();

  // Function
  auto funcName = "main";
  auto funcType = builder.getFunctionType(arg_types, result_types);
  auto func = mlir::FuncOp::create(loc, funcName, funcType);

  // Blocks
  auto &entryBlock = *func.addEntryBlock();
#ifdef printMatrix
  auto external_condition = builder.createBlock(&func.getRegion());
  auto external_loop = builder.createBlock(&func.getRegion());
  auto external_end = builder.createBlock(&func.getRegion());
  auto internal_condition = builder.createBlock(&func.getRegion());
  auto internal_loop = builder.createBlock(&func.getRegion());
  auto internal_end = builder.createBlock(&func.getRegion());
#endif
  builder.setInsertionPointToStart(&entryBlock);

  // Constants
  auto i32_m1 = builder.create<mlir::ConstantIntOp>(loc, -1, 32);
  auto i32_0 = builder.create<mlir::ConstantIntOp>(loc, 0, 32);
  auto i32_1 = builder.create<mlir::ConstantIntOp>(loc, 1, 32);
  auto i64_0 = builder.create<mlir::ConstantIntOp>(loc, 0, 64);
  auto i64_1 = builder.create<mlir::ConstantIntOp>(loc, 1, 64);
  auto i64_2 = builder.create<mlir::ConstantIntOp>(loc, 2, 64);
  auto i64_4 = builder.create<mlir::ConstantIntOp>(loc, 4, 64);
  auto index0 = builder.create<mlir::ConstantIndexOp>(loc, 0);
  auto index1 = builder.create<mlir::ConstantIndexOp>(loc, 1);
  auto index2 = builder.create<mlir::ConstantIndexOp>(loc, 2);
  auto index3 = builder.create<mlir::ConstantIndexOp>(loc, 3);
  auto newLine = builder.create<mlir::LLVM::ConstantOp>(
      loc, llvmI32Ty, builder.getI32IntegerAttr(10));
  auto char1 = builder.create<mlir::LLVM::ConstantOp>(
      loc, llvmI32Ty, builder.getI32IntegerAttr(49));
  auto char0 = builder.create<mlir::LLVM::ConstantOp>(
      loc, llvmI32Ty, builder.getI32IntegerAttr(48));

#ifdef littleMatrix
  auto row = 5;
#else
  auto row = 100000000;
#endif
  auto col = 5;
  auto i32_row = builder.create<mlir::ConstantIntOp>(loc, row, 32);
  auto i32_columns = builder.create<mlir::ConstantIntOp>(loc, col, 32);
  auto i64_row = builder.create<mlir::ConstantIntOp>(loc, row, 64);
  auto i64_columns = builder.create<mlir::ConstantIntOp>(loc, col, 64);
  auto count = builder.create<mlir::ConstantIntOp>(loc, row * col, 64);

  // Device
  auto device = builder.create<mlir::metal::DeviceMakeDefaultOp>(loc);

  // CommandQueue
  auto commandQueue =
      builder.create<mlir::metal::DeviceMakeCommandQueueOp>(loc, device);

  // Buffers
  auto isStorageModeManaged =
      builder.create<mlir::ConstantOp>(loc, builder.getBoolAttr(false));
  auto buffer0 = builder.create<mlir::metal::DeviceMakeBufferOp>(
      loc, device, isStorageModeManaged, count, i64_4);
  auto buffer1 = builder.create<mlir::metal::DeviceMakeBufferOp>(
      loc, device, isStorageModeManaged, i64_2, i64_4);
  auto buffer2 = builder.create<mlir::metal::DeviceMakeBufferOp>(
      loc, device, isStorageModeManaged, count, i64_4);

  // ComputeCommandBuffer
  auto commandBuffer =
      builder.create<mlir::metal::CommandQueueMakeCommandBufferOp>(
          loc, commandQueue, "life", i64_row, i64_columns, i64_1);

  // Fill buffer 0
  {
    auto memRef =
        builder.create<mlir::metal::BufferGetContentsOp>(loc, buffer0, i32Ty);
    builder.create<mlir::StoreOp>(loc, i32_1, memRef, mlir::ValueRange{index0});
    builder.create<mlir::StoreOp>(loc, i32_1, memRef, mlir::ValueRange{index1});
    builder.create<mlir::StoreOp>(loc, i32_1, memRef, mlir::ValueRange{index2});
    builder.create<mlir::StoreOp>(loc, i32_1, memRef, mlir::ValueRange{index3});
  }

  // Fill buffer 1
  {
    auto memRef =
        builder.create<mlir::metal::BufferGetContentsOp>(loc, buffer1, i32Ty);
    builder.create<mlir::StoreOp>(loc, i32_row, memRef,
                                  mlir::ValueRange{index0});
    builder.create<mlir::StoreOp>(loc, i32_columns, memRef,
                                  mlir::ValueRange{index1});
  }

  // Get buffer 2
  auto buffer_result =
      builder.create<mlir::metal::BufferGetContentsOp>(loc, buffer2, i32Ty);

  // Add buffers
  builder.create<mlir::metal::CommandBufferAddBufferOp>(loc, commandBuffer,
                                                        buffer0, i64_0);
  builder.create<mlir::metal::CommandBufferAddBufferOp>(loc, commandBuffer,
                                                        buffer1, i64_1);
  builder.create<mlir::metal::CommandBufferAddBufferOp>(loc, commandBuffer,
                                                        buffer2, i64_2);

  // Commit
  builder.create<mlir::metal::CommandBufferCommitOp>(loc, commandBuffer);

  // Wait
  builder.create<mlir::metal::CommandBufferWaitUntilCompletedOp>(loc,
                                                                 commandBuffer);

#ifdef printMatrix
  {
    mlir::MemRefType memref = mlir::MemRefType::get({}, i32Ty);
    auto i = builder.create<mlir::AllocaOp>(loc, memref);
    auto j = builder.create<mlir::AllocaOp>(loc, memref);
    builder.create<mlir::StoreOp>(loc, i32_m1, i);
    builder.create<mlir::BranchOp>(loc, external_condition);
    {
      // external_condition
      builder.setInsertionPointToStart(external_condition);
      builder.create<mlir::StoreOp>(
          loc,
          builder.create<mlir::AddIOp>(
              loc, builder.create<mlir::LoadOp>(loc, i), i32_1),
          i);
      auto value = builder.create<mlir::CmpIOp>(
          loc, mlir::CmpIPredicate::slt, builder.create<mlir::LoadOp>(loc, i),
          i32_row);
      builder.create<mlir::CondBranchOp>(loc, value, external_loop,
                                         mlir::ValueRange{}, external_end,
                                         mlir::ValueRange{});
    }
    {
      // external_end
      builder.setInsertionPointToStart(external_end);

      builder.create<mlir::metal::ReleaseOp>(loc, device);
      builder.create<mlir::metal::ReleaseOp>(loc, buffer0);
      builder.create<mlir::metal::ReleaseOp>(loc, buffer1);
      builder.create<mlir::metal::ReleaseOp>(loc, buffer2);
      builder.create<mlir::metal::ReleaseOp>(loc, commandQueue);
      builder.create<mlir::metal::ReleaseOp>(loc, commandBuffer);
      builder.create<mlir::ReturnOp>(loc, mlir::ValueRange{i32_0});
    }
    {
      // external_loop
      builder.setInsertionPointToStart(external_loop);
      builder.create<mlir::StoreOp>(loc, i32_m1, j);
      builder.create<mlir::BranchOp>(loc, internal_condition);
    }
    {
      // internal_condition
      builder.setInsertionPointToStart(internal_condition);
      builder.create<mlir::StoreOp>(
          loc,
          builder.create<mlir::AddIOp>(
              loc, builder.create<mlir::LoadOp>(loc, j), i32_1),
          j);
      auto value = builder.create<mlir::CmpIOp>(
          loc, mlir::CmpIPredicate::slt, builder.create<mlir::LoadOp>(loc, j),
          i32_columns);
      builder.create<mlir::CondBranchOp>(
          loc, value, internal_loop, llvm::ArrayRef<mlir::Value>(),
          internal_end, llvm::ArrayRef<mlir::Value>());
    }
    {
      // internal_loop
      builder.setInsertionPointToStart(internal_loop);
      auto product = builder.create<mlir::MulIOp>(
          loc, builder.create<mlir::LoadOp>(loc, i), i32_columns);
      auto indexInt = builder.create<mlir::AddIOp>(
          loc, product, builder.create<mlir::LoadOp>(loc, j));
      auto index = builder.create<mlir::IndexCastOp>(loc, indexInt, indexTy);
      auto value = builder.create<mlir::LoadOp>(loc, buffer_result,
                                                mlir::ValueRange{index});
      auto llvmValue = builder.create<mlir::SelectOp>(
          loc,
          builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, value,
                                       i32_1),
          char1, char0);
      builder.create<mlir::LLVM::CallOp>(loc, putcharF,
                                         mlir::ValueRange{llvmValue});
      builder.create<mlir::BranchOp>(loc, internal_condition);
    }
    {
      // internal_end
      builder.setInsertionPointToStart(internal_end);
      builder.create<mlir::LLVM::CallOp>(loc, putcharF,
                                         mlir::ValueRange{newLine});
      builder.create<mlir::BranchOp>(loc, external_condition);
    }
  }
#else
  builder.create<mlir::metal::ReleaseOp>(loc, device);
  builder.create<mlir::metal::ReleaseOp>(loc, buffer0);
  builder.create<mlir::metal::ReleaseOp>(loc, buffer1);
  builder.create<mlir::metal::ReleaseOp>(loc, buffer2);
  builder.create<mlir::metal::ReleaseOp>(loc, commandQueue);
  builder.create<mlir::metal::ReleaseOp>(loc, commandBuffer);
  builder.create<mlir::ReturnOp>(loc, mlir::ValueRange{i32_0});
#endif

  return func;
}

// -----------------------------------------------------------------------------
// Main

int main() {
  auto driver = Driver{};
  mlir::LLVM::LLVMFuncOp putcharF = driver.insertPutchar();
  driver.addKernel(createKernel(driver.builder(), driver.loc()));
  driver.addOperation(
      createExecuteKernel(driver.builder(), driver.loc(), putcharF));
  driver.verify();
  driver.dump();
  //  driver.canonicalize();
  //  driver.translateToMSL();
  //  driver.translateToLLVM();
}