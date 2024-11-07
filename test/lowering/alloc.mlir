// RUN: %gccjit-opt %s -o %t.mlir -convert-memref-to-gccjit
// RUN: %filecheck --input-file=%t.mlir %s
module @test attributes {
      gccjit.opt_level = #gccjit.opt_level<O3>,
      gccjit.debug_info = false
}
{

  func.func @foo() -> memref<100x100xf32> {
    // CHECK: gccjit.call  builtin @aligned_alloc(%{{[0-9]+}}, %{{[0-9]+}}) : (!gccjit.int<size_t>, !gccjit.int<size_t>) -> !gccjit.ptr<!gccjit.void>
    %a = memref.alloc () : memref<100x100xf32>
    return  %a : memref<100x100xf32>
  }

  func.func @bar(%arg0 : index, %arg1: index) -> memref<?x133x723x?xf32> {
    // CHECK: gccjit.call  builtin @aligned_alloc(%{{[0-9]+}}, %{{[0-9]+}}) : (!gccjit.int<size_t>, !gccjit.int<size_t>) -> !gccjit.ptr<!gccjit.void>
    %a = memref.alloc (%arg0, %arg1) : memref<?x133x723x?xf32>
    return %a : memref<?x133x723x?xf32>
  }

  func.func @baz() -> memref<133x723x1xi128> {
    // CHECK: gccjit.call  builtin @aligned_alloc(%{{[0-9]+}}, %{{[0-9]+}}) : (!gccjit.int<size_t>, !gccjit.int<size_t>) -> !gccjit.ptr<!gccjit.void>
    %a = memref.alloc () {alignment = 128} : memref<133x723x1xi128>
    return %a : memref<133x723x1xi128>
  }
}
