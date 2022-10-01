#!/bin/bash
set -x

llc_bin="llvm-project/build/bin/llc"
metal_translate="build/bin/metal-translate"
metal_opt="build/bin/metal-opt"
runtime_file="build/bin/runtime/Build/Products/Release/libMetalRuntime.a"

metal_file=""$2"/default.metal"
metal_lib=""$2"/default.metallib"
mlir_llvm_file=""$2"/llvm.mlir"
llvm_file=""$2"/llvm.ll"
assembly_file=""$2"/assembly.s"
binary_file=""$2"/$(basename "$1" .mlir).out"

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
clang $assembly_file $runtime_file -o $binary_file -L/usr/lib/swift \
	 -L/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx

# Remove tmp files
rm $metal_file &&
rm $mlir_llvm_file &&
rm $llvm_file &&
rm $assembly_file
