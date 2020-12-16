#include "utility/Driver.h"
namespace ml = mlir::metal;

// -----------------------------------------------------------------------------
// GPU Code

static void createExpressions(mlir::OpBuilder builder, mlir::Location loc) {
  builder.create<ml::ThreadIdOp>(loc, "x");

  auto intValue =
      builder.create<ml::ConstantOp>(loc, builder.getSI32IntegerAttr(10));
  auto boolValue =
      builder.create<ml::ConstantOp>(loc, builder.getBoolAttr(true));

  auto lhs = builder.create<ml::UnaryExpOp>(loc, ml::UnaryExpOperator::notOp,
                                            boolValue);
  auto rhsInt = builder.create<ml::UnaryExpOp>(
      loc, ml::UnaryExpOperator::minusOp, intValue);
  auto rhsBool = builder.create<ml::CastOp>(loc, builder.getI1Type(), rhsInt);

  builder.create<ml::BinaryExpOp>(loc, ml::BinaryExpOperator::eqOp, lhs,
                                  rhsBool);
}

static mlir::metal::KernelOp createKernel(mlir::OpBuilder builder,
                                          mlir::Location loc) {
  llvm::SmallVector<mlir::Type, 3> buffers = {};
  llvm::SmallVector<bool, 3> isAddressSpaceDevice = {};
  auto kernel = builder.create<ml::KernelOp>(loc, "expressions", buffers,
                                             isAddressSpaceDevice);
  builder.setInsertionPointToStart(&kernel.getEntryBlock());

  createExpressions(builder, loc);

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