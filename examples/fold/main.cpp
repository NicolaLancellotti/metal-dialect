#include "utility/Driver.h"
namespace ml = mlir::metal;

// -----------------------------------------------------------------------------
// GPU Code

static void foldCast(mlir::OpBuilder builder, mlir::Location loc) {
  auto value_true =
      builder.create<ml::ConstantOp>(loc, builder.getBoolAttr(true));
  auto value_cast_true_to_bool =
      builder.create<ml::CastOp>(loc, builder.getI1Type(), value_true);
  auto MR =
      ml::MetalMemRefType::get(builder.getContext(), builder.getI1Type(), 1);
  auto memRef = builder.create<ml::AllocaOp>(loc, MR);
  auto index0 =
      builder.create<ml::ConstantOp>(loc, builder.getUI32IntegerAttr(0));
  builder.create<ml::StoreOp>(loc, value_cast_true_to_bool, memRef, index0);
}

static void foldNot(mlir::OpBuilder builder, mlir::Location loc) {
  auto value_true =
      builder.create<ml::ConstantOp>(loc, builder.getBoolAttr(true));
  auto value_not_true = builder.create<ml::UnaryExpOp>(
      loc, ml::UnaryExpOperator::notOp, value_true);

  auto MR =
      ml::MetalMemRefType::get(builder.getContext(), builder.getI1Type(), 1);
  auto memRef = builder.create<ml::AllocaOp>(loc, MR);
  auto index0 =
      builder.create<ml::ConstantOp>(loc, builder.getUI32IntegerAttr(0));
  builder.create<ml::StoreOp>(loc, value_not_true, memRef, index0);
}

static void foldMinus(mlir::OpBuilder builder, mlir::Location loc) {
  auto value_10 =
      builder.create<ml::ConstantOp>(loc, builder.getF16FloatAttr(10));

  auto value_minus_10 = builder.create<ml::UnaryExpOp>(
      loc, ml::UnaryExpOperator::minusOp, value_10);
  auto value_minus_minus_10 = builder.create<ml::UnaryExpOp>(
      loc, ml::UnaryExpOperator::minusOp, value_minus_10);

  auto MR =
      ml::MetalMemRefType::get(builder.getContext(), builder.getF16Type(), 1);
  auto memRef = builder.create<ml::AllocaOp>(loc, MR);
  auto index0 =
      builder.create<ml::ConstantOp>(loc, builder.getUI32IntegerAttr(0));
  builder.create<ml::StoreOp>(loc, value_minus_minus_10, memRef, index0);
}
static mlir::metal::KernelOp createKernel(mlir::OpBuilder builder,
                                          mlir::Location loc) {
  llvm::SmallVector<mlir::Type, 3> buffers = {};
  llvm::SmallVector<bool, 0> isAddressSpaceDevice = {};
  auto kernel =
      builder.create<ml::KernelOp>(loc, "fold", buffers, isAddressSpaceDevice);
  builder.setInsertionPointToStart(&kernel.getEntryBlock());

  foldCast(builder, loc);
  foldNot(builder, loc);
  foldMinus(builder, loc);

  builder.create<ml::ReturnOp>(loc);
  return kernel;
}

// -----------------------------------------------------------------------------
// Main

int main() {
  auto driver = Driver{};
  driver.addKernel(createKernel(driver.builder(), driver.loc()));
  driver.verify();

  driver.canonicalize();

  driver.dumpMSL();
}
