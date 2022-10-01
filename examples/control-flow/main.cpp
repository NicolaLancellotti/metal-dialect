#include "utility/Driver.h"
namespace ml = mlir::metal;

// -----------------------------------------------------------------------------
// GPU Code

static void createIfOp(mlir::OpBuilder builder, mlir::Location loc) {
  auto thenBuilder = [&](mlir::OpBuilder &builder, mlir::Location) {
    builder.create<ml::ConstantOp>(loc, builder.getSI32IntegerAttr(-10));
    builder.create<ml::YieldOp>(loc);
  };
  auto elseBuilder = [&](mlir::OpBuilder &builder, mlir::Location) {
    auto thenBuilder = [&](mlir::OpBuilder &builder, mlir::Location) {
      builder.create<ml::ConstantOp>(loc, builder.getSI32IntegerAttr(-10));
      builder.create<ml::YieldOp>(loc);
    };
    auto condition =
        builder.create<ml::ConstantOp>(loc, builder.getBoolAttr(true));
    builder.create<ml::IfOp>(loc, condition, thenBuilder, nullptr);

    builder.create<ml::ReturnOp>(loc);
  };

  auto condition =
      builder.create<ml::ConstantOp>(loc, builder.getBoolAttr(true));
  builder.create<ml::IfOp>(loc, condition, thenBuilder, elseBuilder);
}

static void createWhile(mlir::OpBuilder builder, mlir::Location loc) {
  auto MR = ml::MetalMemRefType::get(builder.getContext(),
                                     builder.getIntegerType(32, false), 1);
  auto memRef = builder.create<ml::AllocaOp>(loc, MR);
  auto index0 =
      builder.create<ml::ConstantOp>(loc, builder.getUI32IntegerAttr(0));

  auto conditionBuilder = [&](mlir::OpBuilder &builder, mlir::Location) {
    auto one =
        builder.create<ml::ConstantOp>(loc, builder.getUI32IntegerAttr(1));
    auto value = builder.create<ml::GetElementOp>(loc, memRef, index0);
    auto result = builder.create<ml::BinaryExpOp>(
        loc, ml::BinaryExpOperator::subOp, value, one);
    builder.create<ml::StoreOp>(loc, result, memRef, index0);
    auto boolValue =
        builder.create<ml::CastOp>(loc, builder.getI1Type(), value);
    builder.create<ml::YieldWhileOp>(loc, boolValue);
  };
  auto bodyBuilder = [&](mlir::OpBuilder &builder, mlir::Location) {
    builder.create<ml::ReturnOp>(loc);
  };

  builder.create<ml::WhileOp>(loc, conditionBuilder, bodyBuilder);
}

static void createWhile2(mlir::OpBuilder builder, mlir::Location loc) {
  auto conditionBuilder = [&](mlir::OpBuilder &builder, mlir::Location) {
    auto intValue =
        builder.create<ml::ConstantOp>(loc, builder.getSI32IntegerAttr(10));
    auto boolValue =
        builder.create<ml::ConstantOp>(loc, builder.getBoolAttr(true));

    auto lhs = builder.create<ml::UnaryExpOp>(loc, ml::UnaryExpOperator::notOp,
                                              boolValue);
    auto rhsInt = builder.create<ml::UnaryExpOp>(
        loc, ml::UnaryExpOperator::minusOp, intValue);
    auto rhsBool = builder.create<ml::CastOp>(loc, builder.getI1Type(), rhsInt);

    auto condition = builder.create<ml::BinaryExpOp>(
        loc, ml::BinaryExpOperator::eqOp, lhs, rhsBool);

    builder.create<ml::YieldWhileOp>(loc, condition);
  };
  auto bodyBuilder = [&](mlir::OpBuilder &builder, mlir::Location) {
    builder.create<ml::ReturnOp>(loc);
  };

  builder.create<ml::WhileOp>(loc, conditionBuilder, bodyBuilder);
}

static mlir::metal::KernelOp createKernel(mlir::OpBuilder builder,
                                          mlir::Location loc) {
  llvm::SmallVector<mlir::Type, 3> buffers = {};
  llvm::SmallVector<bool, 3> isAddressSpaceDevice = {};
  auto kernel = builder.create<ml::KernelOp>(loc, "controlFlow", buffers,
                                             isAddressSpaceDevice);
  builder.setInsertionPointToStart(&kernel.getEntryBlock());

  createIfOp(builder, loc);
  createWhile(builder, loc);
  createWhile2(builder, loc);

  builder.create<ml::ReturnOp>(loc);
  return kernel;
}

// -----------------------------------------------------------------------------
// Main

int main() {
  auto driver = Driver{};
  driver.addKernel(createKernel(driver.builder(), driver.loc()));
  driver.verify();
  //  driver.canonicalize();
  driver.dump();
  driver.dumpMSL();
}
