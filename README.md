# mlir-gccjit

MLIR dialect for [`libgccjit`](https://gcc.gnu.org/onlinedocs/jit/).

## Prerequisites

In general you need the following tools and libraries to build `mlir-gccjit`:

- A working C++ compiler toolchain that supports C++20 standard.
- [CMake] with minimum version 3.22.
- [Ninja] build system (recommended but not mandatory).
- [LLVM] libraries and development files.
- [MLIR] libraries and development files.
- [libgccjit] libraries and development files.

[CMake]: https://cmake.org/
[Ninja]: https://ninja-build.org/
[LLVM]: https://llvm.org/
[MLIR]: https://mlir.llvm.org/
[libgccjit]: https://gcc.gnu.org/onlinedocs/jit/

For Ubuntu 24.04 (noble) users:

```bash
apt-get install build-essential cmake ninja-build llvm-18-dev libmlir-18-dev libgccjit-13-dev
```

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
