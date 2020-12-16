METAL_BUILD_DIR = ./build
LLVM_BUILD_DIR = ./llvm-project/build
LLVM_INSTALL_DIR = ./llvm-install
MLIR_2_BIN_LIB = ./scripts/mlir2bin-lib.sh
EXAMPLES = ./examples/mlir
EXAMPLE_BUILD_DIR = $(METAL_BUILD_DIR)/bin/examples
JOBS = $(shell sysctl -n hw.logicalcpu)

all:	generate_llvm_project \
		install_llvm \
		generate_metal_project \
		build_metal \
		build_metal_runtime \
		build_examples

clean:
	@echo "Clean"
	@rm -rdf $(LLVM_BUILD_DIR)
	@rm -rdf $(LLVM_INSTALL_DIR)
	@rm -rdf $(METAL_BUILD_DIR)

generate_llvm_project:
	@echo "Generate LLVM Project"
	@mkdir -p $(LLVM_BUILD_DIR)
	@cmake -G Ninja -S ./llvm-project/llvm -B $(LLVM_BUILD_DIR) \
		-DLLVM_ENABLE_PROJECTS=mlir \
		-DLLVM_TARGETS_TO_BUILD="X86" \
		-DCMAKE_BUILD_TYPE=Debug \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DLLVM_INSTALL_UTILS=ON \
		-DLLVM_ENABLE_RTTI=ON \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++ \
		-DCMAKE_INSTALL_PREFIX=$(LLVM_INSTALL_DIR)

install_llvm:
	@echo "Install LLVM"
	@mkdir -p $(LLVM_INSTALL_DIR)
	@cmake --build $(LLVM_BUILD_DIR) --target install -- -j$(JOBS)

generate_metal_project:
	@echo "Generate Metal Project"
	@mkdir -p $(METAL_BUILD_DIR)
	@cmake -G Ninja -S . -B $(METAL_BUILD_DIR) \
		-DMLIR_DIR=$(LLVM_INSTALL_DIR)/lib/cmake/mlir \
		-DLLVM_EXTERNAL_LIT=$(CURDIR)/$(LLVM_BUILD_DIR)/bin/llvm-lit \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++

build_metal:
	@echo "Build Metal"
	@cmake --build $(METAL_BUILD_DIR) -- -j$(JOBS)
	@cmake --build $(METAL_BUILD_DIR) --target check-metal mlir-doc -- -j$(JOBS)		

build_metal_runtime:
	@echo "Build Metal Runtime"
	@xcodebuild -project ./runtime/MetalRuntime/MetalRuntime.xcodeproj \
		-scheme MetalRuntime -derivedDataPath ./build/bin/runtime

build_examples:	build_life5x5_print \
				build_life5x5 \
				build_life100000000x5

build_life5x5_print:
	@echo "Build life5x5_print"
	@mkdir -p $(EXAMPLE_BUILD_DIR)/life5x5_print
	@sh $(MLIR_2_BIN_LIB) $(EXAMPLES)/life5x5_print.mlir $(EXAMPLE_BUILD_DIR)/life5x5_print

build_life5x5:
	@echo "Build life5x5"
	@mkdir -p $(EXAMPLE_BUILD_DIR)/life5x5
	@sh $(MLIR_2_BIN_LIB) $(EXAMPLES)/life5x5.mlir $(EXAMPLE_BUILD_DIR)/life5x5

build_life100000000x5:
	@echo "Build life100000000x5"
	@mkdir -p $(EXAMPLE_BUILD_DIR)/life100000000x5
	@sh $(MLIR_2_BIN_LIB) $(EXAMPLES)/life100000000x5.mlir $(EXAMPLE_BUILD_DIR)/life100000000x5

run_examples:	run_life5x5_print \
				run_life5x5 \
				run_life100000000x5

run_life5x5_print:
	@echo "Run life5x5_print"
	cd $(EXAMPLE_BUILD_DIR)/life5x5_print && ./life5x5_print.out

run_life5x5:
	@echo "Run life5x5"
	cd $(EXAMPLE_BUILD_DIR)/life5x5 && ./life5x5.out

run_life100000000x5:
	@echo "Run life100000000x5"
	cd $(EXAMPLE_BUILD_DIR)/life100000000x5 && ./life100000000x5.out
