// RUN: %gccjit-opt %s -o %t.mlir -convert-memref-to-gccjit
// RUN: %filecheck --input-file=%t.mlir %s
module @test 
{

  func.func @foo() {
    // CHECK: gccjit.call  builtin @aligned_alloc(%{{[0-9]+}}, %{{[0-9]+}}) : (!gccjit.int<size_t>, !gccjit.int<size_t>) -> !gccjit.ptr<!gccjit.void>
    %a = memref.alloc () : memref<100x100xf32>
    return 
  }

  func.func @bar(%arg0 : index, %arg1: index) {
    // CHECK: gccjit.call  builtin @aligned_alloc(%{{[0-9]+}}, %{{[0-9]+}}) : (!gccjit.int<size_t>, !gccjit.int<size_t>) -> !gccjit.ptr<!gccjit.void>
    %a = memref.alloc (%arg0, %arg1) : memref<?x133x723x?xf32>
    return
  }

  func.func @baz() {
    // CHECK: gccjit.call  builtin @aligned_alloc(%{{[0-9]+}}, %{{[0-9]+}}) : (!gccjit.int<size_t>, !gccjit.int<size_t>) -> !gccjit.ptr<!gccjit.void>
    %a = memref.alloc () {alignment = 128} : memref<133x723x1xi128>
    return
  }
}
