# RUN: %python %s | FileCheck %s

from mlir_metal.ir import *
from mlir_metal.dialects import builtin as builtin_d, metal as metal_d

with Context():
    metal_d.register_dialect()
    module = Module.parse(
        """
    %0 = metal.constant 0 : ui32
    """
    )
    # CHECK: %[[C:.*]] = metal.constant 0 : ui32
    print(str(module))
