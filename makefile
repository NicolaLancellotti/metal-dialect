PROJECT_DIR=${shell cd .; pwd}
METAL_BUILD_DIR=./build
LLVM_COMMIT=llvmorg-17.0.1
LLVM_PYTHON_ENV=${HOME}/.venv/mlirdev
LLVM_SRC_DIR=${PROJECT_DIR}/llvm-project/
LLVM_BUILD_DIR=${PROJECT_DIR}/llvm-project/build
RUNTIME_BUILD_DIR=${PROJECT_DIR}/MetalRuntime/.build
MLIR_2_BIN_LIB=${PROJECT_DIR}/scripts/mlir2bin-lib.sh
RUNTIME=${PROJECT_DIR}/MetalRuntime
EXAMPLES=${PROJECT_DIR}/examples/mlir
EXAMPLE_BUILD_DIR=${METAL_BUILD_DIR}/bin/examples

.PHONY: all
all:	llvm \
		metal

.PHONY: help
help:
	@echo "Targets:"
	@sed -nr 's/^.PHONY:(.*)/\1/p' ${MAKEFILE_LIST}

.PHONY: llvm
llvm: 	clone_llvm \
		checkout_llvm \
		generate_llvm_python_env \
		generate_llvm_project \
		build_llvm

.PHONY: metal
metal:	generate_metal_project \
		build_metal \
		build_metal_runtime \
		build_examples

.PHONY: clean_llvm
clean_llvm:
	@echo "Clean LLVM"
	@rm -rdf ${LLVM_BUILD_DIR}

.PHONY: clean_metal
clean_metal:
	@echo "Clean metal"
	@rm -rdf ${METAL_BUILD_DIR}
	@rm -rdf ${RUNTIME_BUILD_DIR}

.PHONY: clone_llvm
clone_llvm:
	@echo "Clone LLVM"
	-git clone https://github.com/llvm/llvm-project.git

.PHONY: checkout_llvm
checkout_llvm:
	@echo "Checkout LLVM"
	@cd ${LLVM_SRC_DIR} && git fetch && git checkout ${LLVM_COMMIT}

.PHONY: generate_llvm_python_env
generate_llvm_python_env:
	@echo "Generate LLVM Python Environment"
	@/usr/bin/python3 -m venv ${LLVM_PYTHON_ENV} && \
		source ${LLVM_PYTHON_ENV}/bin/activate && \
		python -m pip install --upgrade pip && \
		python -m pip install -r ${LLVM_SRC_DIR}/mlir/python/requirements.txt

.PHONY: generate_llvm_project
generate_llvm_project:
	@echo "Generate LLVM Project"
	@cmake -G Ninja -S ${LLVM_SRC_DIR}/llvm -B ${LLVM_BUILD_DIR} \
		-DLLVM_ENABLE_PROJECTS=mlir \
		-DLLVM_TARGETS_TO_BUILD=host \
		-DCMAKE_BUILD_TYPE=Debug \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DLLVM_BUILD_TESTS=ON \
		-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
		-DPython3_EXECUTABLE=${LLVM_PYTHON_ENV}/bin/python3 \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++

.PHONY: build_llvm
build_llvm:
	@echo "build LLVM"
	@cmake --build ${LLVM_BUILD_DIR}

.PHONY: generate_metal_project
generate_metal_project:
	@echo "Generate Metal Project"
	@mkdir -p ${METAL_BUILD_DIR}
	@cmake -G Ninja -S . -B ${METAL_BUILD_DIR} \
		-DMLIR_DIR=${LLVM_BUILD_DIR}/lib/cmake/mlir \
		-DLLVM_EXTERNAL_LIT=${LLVM_BUILD_DIR}/bin/llvm-lit \
		-DCMAKE_BUILD_TYPE=Debug \
		-DPython3_EXECUTABLE=${LLVM_PYTHON_ENV}/bin/python3 \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++

.PHONY: build_metal
build_metal:
	@echo "Build Metal"
	@cmake --build ${METAL_BUILD_DIR}
	@cmake --build ${METAL_BUILD_DIR} --target check-metal mlir-doc

.PHONY: build_metal_runtime
build_metal_runtime:
	@echo "Build Metal Runtime"
	cd ${RUNTIME} && swift build -c release

.PHONY: build_examples
build_examples:	build_metal_runtime_example \
				build_life5x5_print \
				build_life5x5 \
				build_life100000000x5

.PHONY: build_metal_runtime_example
build_metal_runtime_example:
	@echo "Build Metal Runtime"
	@xcodebuild -project ./examples/MetalRuntimeExample/MetalRuntimeExample.xcodeproj \
		-scheme MetalRuntimeExample

.PHONY: build_life5x5_print
build_life5x5_print:
	@echo "Build life5x5_print"
	@mkdir -p ${EXAMPLE_BUILD_DIR}/life5x5_print
	@sh ${MLIR_2_BIN_LIB} ${EXAMPLES}/life5x5_print.mlir ${EXAMPLE_BUILD_DIR}/life5x5_print ${LLVM_BUILD_DIR}

.PHONY: build_life5x5
build_life5x5:
	@echo "Build life5x5"
	@mkdir -p ${EXAMPLE_BUILD_DIR}/life5x5
	@sh ${MLIR_2_BIN_LIB} ${EXAMPLES}/life5x5.mlir ${EXAMPLE_BUILD_DIR}/life5x5 ${LLVM_BUILD_DIR}

.PHONY: build_life100000000x5
build_life100000000x5:
	@echo "Build life100000000x5"
	@mkdir -p ${EXAMPLE_BUILD_DIR}/life100000000x5
	@sh ${MLIR_2_BIN_LIB} ${EXAMPLES}/life100000000x5.mlir ${EXAMPLE_BUILD_DIR}/life100000000x5 ${LLVM_BUILD_DIR}

.PHONY: run_examples
run_examples:	run_life5x5_print \
				run_life5x5 \
				run_life100000000x5

.PHONY: run_life5x5_print
run_life5x5_print:
	@echo "Run life5x5_print"
	cd ${EXAMPLE_BUILD_DIR}/life5x5_print && ./life5x5_print

.PHONY: run_life5x5
run_life5x5:
	@echo "Run life5x5"
	cd ${EXAMPLE_BUILD_DIR}/life5x5 && ./life5x5

.PHONY: run_life100000000x5
run_life100000000x5:
	@echo "Run life100000000x5"
	cd ${EXAMPLE_BUILD_DIR}/life100000000x5 && ./life100000000x5

.PHONY: format
format:
	@find . -name "*.cpp" -or -name "*.h" | xargs clang-format -i
