// RUN: %gccjit-opt %s -convert-memref-to-gccjit  -convert-memref-to-gccjit -convert-func-to-gccjit -reconcile-unrealized-casts -o %t.mlir -mlir-print-debuginfo
// RUN: %gccjit-translate %t.mlir -o %t.gimple -mlir-to-gccjit-gimple 
// RUN: %filecheck --input-file=%t.gimple %s --check-prefix=CHECK-GIMPLE
module @test  attributes {
  gccjit.opt_level = #gccjit.opt_level<O3>,
  gccjit.debug_info = false
}
{
  func.func @foo(%arg0: memref<100x100xf32>) -> memref<100x100xf32> {
    // CHECK-GIMPLE: %0 = %arg0;
    // CHECK-GIMPLE: %1 = (struct memref<100x100xf32>) {.base=%0.base, .aligned=bitcast(__builtin_assume_aligned ((bitcast(%0.aligned, void *)), %0.offset), float *), .offset=%0.offset, .sizes=%0.sizes, .strides=%0.strides};
    // CHECK-GIMPLE: return %1;
    memref.assume_alignment %arg0, 128 : memref<100x100xf32>
    return %arg0 : memref<100x100xf32>
  }
}
