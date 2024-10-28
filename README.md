# mlir-gccjit

MLIR dialect for [`libgccjit`](https://gcc.gnu.org/onlinedocs/jit/).

## Prerequisites

In general you need the following tools and libraries to build `mlir-gccjit`:

- A working C++ compiler toolchain that supports C++17 standard.
- [CMake] with minimum version 3.22.
- [Ninja] build system (recommended but not mandatory).
- [LLVM] libraries and development files. For now we have only tested LLVM-18.
- [MLIR] libraries and development files. For now we have only tested MLIR-18.
- [libgccjit] libraries and development files. For now we have only tested
  libgccjit-14.

[CMake]: https://cmake.org/
[Ninja]: https://ninja-build.org/
[LLVM]: https://llvm.org/
[MLIR]: https://mlir.llvm.org/
[libgccjit]: https://gcc.gnu.org/onlinedocs/jit/

For Ubuntu 24.04 (noble) users:

```bash
apt-get install build-essential cmake ninja-build llvm-18-dev llvm-18-tools libmlir-18-dev libgccjit-14-dev mlir-18-tools
```

Additionally, you need the [`lit` tool] to run tests. You can install the tool
via `pip`:

```bash
pip install lit
```

[`lit` tool]: https://llvm.org/docs/CommandGuide/lit.html

## Build

Clone the repository:

```bash
git clone https://github.com/Lancern/mlir-gccjit.git
cd mlir-gccjit
```

Create a build directory:

```bash
mkdir build
cd build
```

Build with CMake:

```bash
cmake -G Ninja ..
cmake --build .
```

## License

This project is licensed under [Apache License 2.0](./LICENSE).
