import os
from lit.formats import ShTest

config.name = "MLIR GCCJIT Dialect"
config.test_format = ShTest(True)

config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.mlir_gccjit_build_root, "test")

config.substitutions.append((r"%filecheck", config.file_check_exe))
config.substitutions.append(
    (r"%gccjit-opt", os.path.join(config.mlir_gccjit_build_root, "bin/gccjit-opt"))
)
config.substitutions.append(
    (
        r"%gccjit-translate",
        os.path.join(config.mlir_gccjit_build_root, "bin/gccjit-translate"),
    )
)
