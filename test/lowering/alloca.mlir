// RUN: %gccjit-opt %s -o %t.mlir -convert-memref-to-gccjit
// RUN: %filecheck --input-file=%t.mlir %s
module @test 
{

  func.func @foo() {
    // CHECK: gccjit.call  builtin @__builtin_alloca(%{{[0-9]+}}) : (!gccjit.int<size_t>) -> !gccjit.ptr<!gccjit.void>
    %a = memref.alloca () : memref<100x100xf32>
    return 
  }

  func.func @bar(%arg0 : index) {
    // // CHECK: gccjit.call  builtin @__builtin_alloca(%{{[0-9]+}}) : (!gccjit.int<size_t>) -> !gccjit.ptr<!gccjit.void>
    %a = memref.alloca (%arg0) : memref<133x723x?xf32>
    return
  }

  func.func @baz(%arg0 : index) {
    // CHECK: gccjit.call  builtin @__builtin_alloca_with_align(%{{[0-9]+}}, %{{[0-9]+}}) : (!gccjit.int<size_t>, !gccjit.int<size_t>) -> !gccjit.ptr<!gccjit.void>
    %a = memref.alloca (%arg0) {alignment = 128} : memref<133x723x?xf32>
    return
  }
}
