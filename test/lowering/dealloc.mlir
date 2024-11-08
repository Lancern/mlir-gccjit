// RUN: %gccjit-opt %s \
// RUN:     -lower-affine \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-arith-to-gccjit \
// RUN:     -convert-memref-to-gccjit \
// RUN:     -convert-func-to-gccjit \
// RUN:     -reconcile-unrealized-casts -mlir-print-debuginfo -o %t.mlir

// RUN: %gccjit-translate %t.mlir -mlir-to-gccjit-executable -o %t.exe && chmod +x %t.exe
// RUN: %t.exe
module @test attributes {
    gccjit.cmdline_options = ["-fsanitize=address"],
    gccjit.driver_options = ["-fsanitize=address"],
    gccjit.debug_info = true
}
{
  func.func @main() {
    %arg0 = arith.constant 133 : index
    %a = memref.alloc (%arg0) : memref<133x723x?xf32>
    memref.dealloc %a : memref<133x723x?xf32>
    return
  }
}
