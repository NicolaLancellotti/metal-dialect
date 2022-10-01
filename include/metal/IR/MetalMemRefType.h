//===--- MetalMemRefType.h - MLIR lowering of Metal MemRef Type -*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_METALMEMREFTYPE_H
#define METAL_METALMEMREFTYPE_H

#include "mlir/IR/Types.h"

namespace mlir {
namespace metal {

namespace detail {

struct MetalMemRefTypeStorage : public TypeStorage {
  uint32_t _size;
  Type _type;

  MetalMemRefTypeStorage(uint32_t size, Type type) : _size(size), _type(type) {}

  using KeyTy = std::tuple<uint32_t, Type>;

  bool operator==(const KeyTy &key) const {
    return key == std::make_tuple(_size, _type);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
  }

  static MetalMemRefTypeStorage *construct(TypeStorageAllocator &allocator,
                                           const KeyTy &key) {
    return new (allocator.allocate<MetalMemRefTypeStorage>())
        MetalMemRefTypeStorage(std::get<0>(key), std::get<1>(key));
  }
};

} // end namespace detail

class MetalMemRefType
    : public mlir::Type::TypeBase<MetalMemRefType, mlir::Type,
                                  detail::MetalMemRefTypeStorage> {
public:
  using Base::Base;

  static MetalMemRefType get(mlir::MLIRContext *ctx, Type type, uint32_t size) {
    return Base::get(ctx, size, type);
  }

  Type getType() { return getImpl()->_type; }

  uint32_t getSize() { return getImpl()->_size; }
};

} // end namespace metal
} // end namespace mlir

#endif // METAL_METALMEMREFTYPE_H
