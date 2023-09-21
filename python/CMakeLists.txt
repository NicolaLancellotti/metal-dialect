include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_metal`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_metal.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(MetalPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MetalPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_metal"
  TD_FILE dialects/MetalOps.td
  SOURCES
    dialects/metal.py
  DIALECT_NAME metal)

declare_mlir_python_extension(MetalPythonSources.Extension
  MODULE_NAME _metalDialects
  ADD_TO_PARENT MetalPythonSources
  SOURCES
    MetalExtension.cpp
  EMBED_CAPI_LINK_LIBS
    MetalCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(MetalPythonCAPI
  INSTALL_COMPONENT MetalPythonModules
  INSTALL_DESTINATION python_packages/metal/mlir_metal/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/metal/mlir_metal/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MetalPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(MetalPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/metal/mlir_metal"
  INSTALL_PREFIX "python_packages/metal/mlir_metal"
  DECLARED_SOURCES
    MetalPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    MetalPythonCAPI
  )
