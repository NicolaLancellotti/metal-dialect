# _____________________________________________________________________________
# Parameters

METAL_PRESET=debug

LLVM_COMMIT=llvmorg-18.1.0
LLVM_PRESET=release
LLVM_BUILD_DIR=${LLVM_SRC_DIR}/build/${LLVM_PRESET}
LLVM_PYTHON_ENV=${HOME}/.venv/mlirdev

# _____________________________________________________________________________
# Paths

PROJECT_DIR=${shell cd .; pwd}
METAL_BUILD_DIR=./build
MLIR_2_BIN_LIB=${PROJECT_DIR}/scripts/mlir2bin-lib.sh

RUNTIME=${PROJECT_DIR}/MetalRuntime
RUNTIME_BUILD_DIR=${PROJECT_DIR}/MetalRuntime/.build

EXAMPLES=${PROJECT_DIR}/examples/mlir
EXAMPLE_BUILD_DIR=${PROJECT_DIR}/build/${METAL_PRESET}/bin/examples

LLVM_SRC_DIR=${PROJECT_DIR}/llvm-project/

# _____________________________________________________________________________
# Targets

.PHONY: all
all:	llvm-all \
		metal-all

.PHONY: help
help:
	@echo "Targets:"
	@sed -nr 's/^.PHONY:(.*)/\1/p' ${MAKEFILE_LIST}

define format
	@find ${1} -name "*.cpp" -or -name "*.h" | xargs clang-format -i
endef

.PHONY: format
format:
	@echo "Format"
	@$(call format, examples)
	@$(call format, include)
	@$(call format, lib)
	@$(call format, metal-opt)
	@$(call format, metal-translate)
	@$(call format, MetalRuntime)
	@$(call format, test)

# _____________________________________________________________________________
# Targets - LLVM

.PHONY: llvm-all
llvm-all: 	llvm-clone \
			llvm-checkout \
			llvm-generate-python-env \
			llvm-generate-project \
			llvm-build

.PHONY: llvm-clean
llvm-clean:
	@echo "LLVM - Clean"
	@rm -rdf ${LLVM_BUILD_DIR}

.PHONY: llvm-clone
llvm-clone:
	@echo "LLVM - Clone"
	-git clone https://github.com/llvm/llvm-project.git

.PHONY: llvm-checkout
llvm-checkout:
	@echo "LLVM - Checkout"
	@cd ${LLVM_SRC_DIR} && git fetch && git checkout ${LLVM_COMMIT}

.PHONY: llvm-generate-python-env
llvm-generate-python-env:
	@echo "LLVM - Generate Python Environment"
	@/usr/bin/python3 -m venv ${LLVM_PYTHON_ENV} && \
		source ${LLVM_PYTHON_ENV}/bin/activate && \
		python -m pip install --upgrade pip && \
		python -m pip install -r ${LLVM_SRC_DIR}/mlir/python/requirements.txt

.PHONY: llvm-generate-project
llvm-generate-project:
	@echo "LLVM - Generate Project"
	@cmake -G Ninja -S ${LLVM_SRC_DIR}/llvm -B ${LLVM_BUILD_DIR} \
		-DLLVM_ENABLE_PROJECTS=mlir \
		-DLLVM_TARGETS_TO_BUILD=host \
		-DCMAKE_BUILD_TYPE=${LLVM_PRESET} \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DLLVM_BUILD_TESTS=ON \
		-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
		-DPython3_EXECUTABLE=${LLVM_PYTHON_ENV}/bin/python3 \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++

.PHONY: llvm-build
llvm-build:
	@echo "LLVM - Build"
	@cmake --build ${LLVM_BUILD_DIR}

# _____________________________________________________________________________
# Targets - Metal

.PHONY: metal-all
metal-all:	metal-generate-presets \
			metal-generate-project \
			metal-copy-compile-commands \
			metal-build-mlir \
			metal-build-runtime \
			metal-build-examples

.PHONY: metal-clean
metal-clean:
	@echo "Metal - Clean"
	@rm -rdf ${METAL_BUILD_DIR}
	@rm -rdf ${RUNTIME_BUILD_DIR}

.PHONY: metal-generate-presets
metal-generate-presets:
	@echo "Metal - Generate Presets"
	@echo $$CMAKE_PRESETS_TEMPLATE > ./CMakeUserPresets.json

.PHONY: metal-generate-project
metal-generate-project:
	@echo "Metal - Generate Project"
	@cmake -S ${PROJECT_DIR} --preset ${METAL_PRESET}

.PHONY: metal-copy-compile-commands
metal-copy-compile-commands:
	@echo "Metal - Copy compile_commands.json"
	@cp ${PROJECT_DIR}/build/${METAL_PRESET}/compile_commands.json  ${PROJECT_DIR}/build

.PHONY: metal-build-mlir
metal-build-mlir:
	@echo "Metal - Build"
	@cmake --build ${PROJECT_DIR}/build/${METAL_PRESET}
	@cmake --build ${PROJECT_DIR}/build/${METAL_PRESET} --target check-metal mlir-doc

.PHONY: metal-build-runtime
metal-build-runtime:
	@echo "Metal - Build Runtime"
	cd ${RUNTIME} && swift build -c release

# _____________________________________________________________________________
# Targets - Metal - Examples

.PHONY: metal-build-examples
metal-build-examples:	metal-build-runtime-example \
						metal-build-example-life5x5_print \
						metal-build-example-life5x5 \
						metal-build-example-life100000000x5

.PHONY: metal-build-runtime-example
metal-build-runtime-example:
	@echo "Metal - Build Runtime Example"
	@xcodebuild -project ./examples/MetalRuntimeExample/MetalRuntimeExample.xcodeproj \
		-scheme MetalRuntimeExample

.PHONY: metal-build-example-life5x5_print
metal-build-example-life5x5_print:
	@echo "Metal - Build Example life5x5_print"
	@mkdir -p ${EXAMPLE_BUILD_DIR}/life5x5_print
	@sh ${MLIR_2_BIN_LIB} ${EXAMPLES}/life5x5_print.mlir ${EXAMPLE_BUILD_DIR}/life5x5_print ${LLVM_BUILD_DIR} ${METAL_PRESET}

.PHONY: metal-build-example-life5x5
metal-build-example-life5x5:
	@echo "Metal - Build Example life5x5"
	@mkdir -p ${EXAMPLE_BUILD_DIR}/life5x5
	@sh ${MLIR_2_BIN_LIB} ${EXAMPLES}/life5x5.mlir ${EXAMPLE_BUILD_DIR}/life5x5 ${LLVM_BUILD_DIR} ${METAL_PRESET}

.PHONY: metal-build-example-life100000000x5
metal-build-example-life100000000x5:
	@echo "Metal - Build Example life100000000x5"
	@mkdir -p ${EXAMPLE_BUILD_DIR}/life100000000x5
	@sh ${MLIR_2_BIN_LIB} ${EXAMPLES}/life100000000x5.mlir ${EXAMPLE_BUILD_DIR}/life100000000x5 ${LLVM_BUILD_DIR} ${METAL_PRESET}

.PHONY: metal-run-examples
metal-run-examples:	metal-run-example-life5x5_print \
					metal-run-example-life5x5 \
					metal-run-example-life100000000x5

.PHONY: metal-run-example-life5x5_print
metal-run-example-life5x5_print:
	@echo "Metal - Run Example life5x5_print"
	cd ${EXAMPLE_BUILD_DIR}/life5x5_print && ./life5x5_print

.PHONY: metal-run-example-life5x5
metal-run-example-life5x5:
	@echo "Metal - Run Example life5x5"
	cd ${EXAMPLE_BUILD_DIR}/life5x5 && ./life5x5

.PHONY: metal-run-example-life100000000x5
metal-run-example-life100000000x5:
	@echo "Metal - Run Example life100000000x5"
	cd ${EXAMPLE_BUILD_DIR}/life100000000x5 && ./life100000000x5

# _____________________________________________________________________________
# Presets

define CMAKE_PRESETS_TEMPLATE
{
    "version": 3,
    "configurePresets": [
        {
            "name": "default",
            "hidden": true,
            "displayName": "Default configure preset",
            "description": "Default configure preset",
            "generator": "Ninja",
            "binaryDir": "./build/$${presetName}",
            "cacheVariables": {
				"MLIR_DIR": "${LLVM_BUILD_DIR}/lib/cmake/mlir",
				"LLVM_EXTERNAL_LIT": "${LLVM_BUILD_DIR}/bin/llvm-lit",
				"Python3_EXECUTABLE": "${LLVM_PYTHON_ENV}/bin/python3",
				"CMAKE_C_COMPILER": "/usr/bin/clang",
                "CMAKE_CXX_COMPILER": "/usr/bin/clang++",
				"CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
            }
        },
 		{
            "name": "debug",
            "inherits": "default",
            "displayName": "Debug",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Debug"
            }
        },
        {
            "name": "release",
            "inherits": "default",
            "displayName": "Release",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release"
            }
        },
        {
            "name": "relWithDebInfo",
            "inherits": "default",
            "displayName": "RelWithDebInfo",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "RelWithDebInfo"
            }
        }
    ]
}
endef
export CMAKE_PRESETS_TEMPLATE
