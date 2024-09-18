#!/bin/bash
set -x

input=$1
output=$2
llvm_build_dir=$3
preset=$4

llc_bin=$llvm_build_dir/bin/llc
metal_translate=build/${preset}/bin/metal-translate
metal_opt=build/${preset}/bin/metal-opt
runtime_file=MetalRuntime/.build/release/libMetalRuntime.a

metal_file=$output/default.metal
metal_lib=$output/default.metallib
mlir_llvm_file=$output/llvm.mlir
llvm_file=$output/llvm.ll
assembly_file=$output/assembly.s
binary_file=$output/$(basename "$1" .mlir)

# mlir to metal shading language
$metal_translate $1 --mlir-to-msl 1> $metal_file &&

# make metal library
xcrun -sdk macosx metal $metal_file -o $metal_lib &&

# mlir to mlir-llvm
$metal_opt $1 --convert-metal-to-llvm 1> $mlir_llvm_file &&

# mlir-llvm to llvm
$metal_translate $mlir_llvm_file --mlir-to-llvmir 1> $llvm_file  &&

# llvm to assembly
$llc_bin $llvm_file -o $assembly_file  &&

# Compile & Link
clang $assembly_file $runtime_file \
  -L/usr/lib/swift \
  -L$(xcode-select -p)/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx \
  -framework CoreGraphics \
  -o $binary_file

# Remove tmp files
rm $metal_file &&
rm $mlir_llvm_file &&
rm $llvm_file &&
rm $assembly_file
