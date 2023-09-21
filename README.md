# MLIR metal dialect
## Getting Started
- Install [Xcode](https://developer.apple.com/xcode/)
- Change the path to the active Xcode developer directory:
```
sudo xcode-select -s <path to Xcode.app>/Contents/Developer
```
- Install CMake:
```
brew install cmake
```
- Install Ninja:
```
brew install ninja
```
- Build the LLVM project, the metal project and the examples:
```
make all
```
or, if you have already built LLVM in `$LLVM_BUILD_DIR`, build the metal project and the examples:
```
make metal LLVM_BUILD_DIR=$LLVM_BUILD_DIR
```
- Run the examples
```
make run_examples                                   # Default LLVM build
make run_examples LLVM_BUILD_DIR=$LLVM_BUILD_DIR    # Custom LLVM build
```
