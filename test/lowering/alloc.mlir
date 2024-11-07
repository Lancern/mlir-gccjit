// RUN: %gccjit-opt %s -o %t.mlir -convert-memref-to-gccjit
// RUN: %filecheck --input-file=%t.mlir %s
module @test 
{

  func.func @foo() {
    // CHECK: gccjit.call  builtin @malloc(%{{[0-9]+}}) : (!gccjit.int<size_t>) -> !gccjit.ptr<!gccjit.void>
    %a = memref.alloc () : memref<100x100xf32>
    return 
  }

  func.func @bar(%arg0 : index, %arg1: index) {
    // CHECK: gccjit.call  builtin @malloc(%{{[0-9]+}}) : (!gccjit.int<size_t>) -> !gccjit.ptr<!gccjit.void>
    %a = memref.alloc (%arg0, %arg1) : memref<?x133x723x?xf32>
    return
  }

  func.func @baz() {
    // CHECK: %[[V6:[0-9]+]] = gccjit.sizeof !gccjit.int<uint128_t> : <size_t>
    // CHECK: %[[V7:[0-9]+]] = gccjit.binary mult(%[[V6]] : !gccjit.int<size_t>, %{{[0-9]+}} : !gccjit.int<size_t>) : !gccjit.int<size_t>
    // CHECK: %[[V8:[0-9]+]] = gccjit.const #gccjit.int<128> : !gccjit.int<size_t>
    // CHECK: %[[V9:[0-9]+]] = gccjit.binary plus(%[[V7]] : !gccjit.int<size_t>, %[[V8]] : !gccjit.int<size_t>) : !gccjit.int<size_t>
    // CHECK: %[[V10:[0-9]+]] = gccjit.call  builtin @malloc(%[[V9]]) : (!gccjit.int<size_t>) -> !gccjit.ptr<!gccjit.void>
    // CHECK: %[[V11:[0-9]+]] = gccjit.bitcast %[[V10]] : !gccjit.ptr<!gccjit.void> to !gccjit.int<size_t>
    // CHECK: %[[V12:[0-9]+]] = gccjit.const #gccjit.int<1> : !gccjit.int<size_t>
    // CHECK: %[[V13:[0-9]+]] = gccjit.binary minus(%[[V8]] : !gccjit.int<size_t>, %[[V12]] : !gccjit.int<size_t>) : !gccjit.int<size_t>
    // CHECK: %[[V14:[0-9]+]] = gccjit.binary plus(%[[V11]] : !gccjit.int<size_t>, %[[V13]] : !gccjit.int<size_t>) : !gccjit.int<size_t>
    // CHECK: %[[V15:[0-9]+]] = gccjit.binary modulo(%[[V14]] : !gccjit.int<size_t>, %[[V8]] : !gccjit.int<size_t>) : !gccjit.int<size_t>
    // CHECK: %[[V16:[0-9]+]] = gccjit.binary minus(%[[V14]] : !gccjit.int<size_t>, %[[V15]] : !gccjit.int<size_t>) : !gccjit.int<size_t>
    // CHECK: %[[V17:[0-9]+]] = gccjit.bitcast %[[V16]] : !gccjit.int<size_t> to !gccjit.ptr<!gccjit.int<uint128_t>>
    %a = memref.alloc () {alignment = 128} : memref<133x723x1xi128>
    return
  }
}
