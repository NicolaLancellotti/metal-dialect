#include "utility/Driver.h"
namespace ml = mlir::metal;

// -----------------------------------------------------------------------------
// GPU Code

static void createStoreBuffer(mlir::OpBuilder builder, mlir::Location loc,
                              mlir::Value buffer) {
  auto index_0 =
      builder.create<ml::ConstantOp>(loc, builder.getUI32IntegerAttr(0));
  auto value_0 =
      builder.create<ml::ConstantOp>(loc, builder.getF32FloatAttr(0));
  builder.create<ml::StoreOp>(loc, value_0, buffer, index_0);
}

static void createAllocaStoreLoad(mlir::OpBuilder builder, mlir::Location loc) {
  auto MR =
      ml::MetalMemRefType::get(builder.getContext(), builder.getF32Type(), 10);
  auto memRef = builder.create<ml::AllocaOp>(loc, MR);

  auto index1 =
      builder.create<ml::ConstantOp>(loc, builder.getUI32IntegerAttr(1));
  auto value1 =
      builder.create<ml::ConstantOp>(loc, builder.getF32FloatAttr(1.0));
  builder.create<ml::StoreOp>(loc, value1, memRef, index1);

  auto index2 =
      builder.create<ml::ConstantOp>(loc, builder.getUI32IntegerAttr(2));
  auto value = builder.create<ml::GetElementOp>(loc, memRef, index2);
  builder.create<ml::StoreOp>(loc, value, memRef, index1);
}

static mlir::metal::KernelOp createKernel(mlir::OpBuilder builder,
                                          mlir::Location loc) {
  auto f32 = builder.getF32Type();
  llvm::SmallVector<mlir::Type, 3> buffers = {f32};
  llvm::SmallVector<bool, 3> isAddressSpaceDevice = {true};
  auto kernel = builder.create<ml::KernelOp>(loc, "memory", buffers,
                                             isAddressSpaceDevice);
  builder.setInsertionPointToStart(&kernel.getEntryBlock());

  createStoreBuffer(builder, loc, kernel.getBuffer(0));
  createAllocaStoreLoad(builder, loc);

  builder.create<ml::ReturnOp>(loc);
  return kernel;
}

// -----------------------------------------------------------------------------
// Main

int main() {
  auto driver = Driver{};
  driver.addKernel(createKernel(driver.builder(), driver.loc()));
  driver.verify();

  driver.dump();
  driver.dumpMSL();
}
