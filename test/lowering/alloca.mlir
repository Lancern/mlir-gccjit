// RUN: %gccjit-opt %s -o %t.mlir -convert-memref-to-gccjit
// RUN: %filecheck --input-file=%t.mlir %s
module @test 
{

  func.func @foo() {
    // CHECK: %[[V0:[0-9]+]] = gccjit.expr  {
    // CHECK:   %[[V1:[0-9]+]] = gccjit.const #gccjit.int<100> : !gccjit.int<size_t>
    // CHECK:   %[[V2:[0-9]+]] = gccjit.const #gccjit.int<100> : !gccjit.int<size_t>
    // CHECK:   %[[V3:[0-9]+]] = gccjit.const #gccjit.int<1> : !gccjit.int<size_t>
    // CHECK:   %[[V4:[0-9]+]] = gccjit.const #gccjit.int<10000> : !gccjit.int<size_t>
    // CHECK:   %[[V5:[0-9]+]] = gccjit.sizeof !gccjit.fp<float> : <size_t>
    // CHECK:   %[[V6:[0-9]+]] = gccjit.binary mult(%[[V5]] : !gccjit.int<size_t>, %[[V4]] : !gccjit.int<size_t>) : !gccjit.int<size_t>
    // CHECK:   %[[V7:[0-9]+]] = gccjit.call  builtin @alloca(%[[V6]]) : (!gccjit.int<size_t>) -> !gccjit.ptr<!gccjit.void>
    // CHECK:   %[[V8:[0-9]+]] = gccjit.bitcast %[[V7]] : !gccjit.ptr<!gccjit.void> to !gccjit.ptr<!gccjit.fp<float>>
    // CHECK:   %[[V9:[0-9]+]] = gccjit.new_array <!gccjit.int<size_t>, 2>[%[[V1]] : !gccjit.int<size_t>, %[[V2]] : !gccjit.int<size_t>]
    // CHECK:   %[[V10:[0-9]+]] = gccjit.new_array <!gccjit.int<size_t>, 2>[%[[V2]] : !gccjit.int<size_t>, %[[V3]] : !gccjit.int<size_t>]
    // CHECK:   %[[V11:[0-9]+]] = gccjit.const #gccjit.int<0> : !gccjit.int<size_t>
    // CHECK:   %[[V12:[0-9]+]] = gccjit.new_struct [0, 1, 2, 3, 4][%[[V8]], %[[V8]], %[[V11]], %[[V9]], %[[V10]]] : (!gccjit.ptr<!gccjit.fp<float>>, !gccjit.ptr<!gccjit.fp<float>>, !gccjit.int<size_t>, !gccjit.array<!gccjit.int<size_t>, 2>, !gccjit.array<!gccjit.int<size_t>, 2>) -> !gccjit.struct<"memref<100x100xf32>" {#gccjit.field<"base" !gccjit.ptr<!gccjit.fp<float>>>, #gccjit.field<"aligned" !gccjit.ptr<!gccjit.fp<float>>>, #gccjit.field<"offset" !gccjit.int<size_t>>, #gccjit.field<"sizes" !gccjit.array<!gccjit.int<size_t>, 2>>, #gccjit.field<"strides" !gccjit.array<!gccjit.int<size_t>, 2>>}>
    // CHECK:   gccjit.return %[[V12]] : !gccjit.struct<"memref<100x100xf32>" {#gccjit.field<"base" !gccjit.ptr<!gccjit.fp<float>>>, #gccjit.field<"aligned" !gccjit.ptr<!gccjit.fp<float>>>, #gccjit.field<"offset" !gccjit.int<size_t>>, #gccjit.field<"sizes" !gccjit.array<!gccjit.int<size_t>, 2>>, #gccjit.field<"strides" !gccjit.array<!gccjit.int<size_t>, 2>>}>
    // CHECK: } : !gccjit.struct<"memref<100x100xf32>" {#gccjit.field<"base" !gccjit.ptr<!gccjit.fp<float>>>, #gccjit.field<"aligned" !gccjit.ptr<!gccjit.fp<float>>>, #gccjit.field<"offset" !gccjit.int<size_t>>, #gccjit.field<"sizes" !gccjit.array<!gccjit.int<size_t>, 2>>, #gccjit.field<"strides" !gccjit.array<!gccjit.int<size_t>, 2>>}>
    %a = memref.alloca () : memref<100x100xf32>
    return 
  }

  func.func @bar(%arg0 : index) {
    // CHECK: %[[V0:[0-9]+]] = builtin.unrealized_conversion_cast %{{[0-9a-z]+}} : index to !gccjit.int<size_t>
    // CHECK: %[[V1:[0-9]+]] = gccjit.expr  {
    // CHECK:   %[[V2:[0-9]+]] = gccjit.const #gccjit.int<133> : !gccjit.int<size_t>
    // CHECK:   %[[V3:[0-9]+]] = gccjit.const #gccjit.int<723> : !gccjit.int<size_t>
    // CHECK:   %[[V4:[0-9]+]] = gccjit.const #gccjit.int<1> : !gccjit.int<size_t>
    // CHECK:   %[[V5:[0-9]+]] = gccjit.binary mult(%[[V0]] : !gccjit.int<size_t>, %[[V3]] : !gccjit.int<size_t>) : !gccjit.int<size_t>
    // CHECK:   %[[V6:[0-9]+]] = gccjit.binary mult(%[[V5]] : !gccjit.int<size_t>, %[[V2]] : !gccjit.int<size_t>) : !gccjit.int<size_t>
    // CHECK:   %[[V7:[0-9]+]] = gccjit.sizeof !gccjit.fp<float> : <size_t>
    // CHECK:   %[[V8:[0-9]+]] = gccjit.binary mult(%[[V7]] : !gccjit.int<size_t>, %[[V6]] : !gccjit.int<size_t>) : !gccjit.int<size_t>
    // CHECK:   %[[V9:[0-9]+]] = gccjit.call  builtin @alloca(%[[V8]]) : (!gccjit.int<size_t>) -> !gccjit.ptr<!gccjit.void>
    // CHECK:   %[[V10:[0-9]+]] = gccjit.bitcast %[[V9]] : !gccjit.ptr<!gccjit.void> to !gccjit.ptr<!gccjit.fp<float>>
    // CHECK:   %[[V11:[0-9]+]] = gccjit.new_array <!gccjit.int<size_t>, 3>[%[[V2]] : !gccjit.int<size_t>, %[[V3]] : !gccjit.int<size_t>, %[[V0]] : !gccjit.int<size_t>]
    // CHECK:   %[[V12:[0-9]+]] = gccjit.new_array <!gccjit.int<size_t>, 3>[%[[V5]] : !gccjit.int<size_t>, %[[V0]] : !gccjit.int<size_t>, %[[V4]] : !gccjit.int<size_t>]
    // CHECK:   %[[V13:[0-9]+]] = gccjit.const #gccjit.int<0> : !gccjit.int<size_t>
    // CHECK:   %[[V14:[0-9]+]] = gccjit.new_struct [0, 1, 2, 3, 4][%[[V10]], %[[V10]], %[[V13]], %[[V11]], %[[V12]]] : (!gccjit.ptr<!gccjit.fp<float>>, !gccjit.ptr<!gccjit.fp<float>>, !gccjit.int<size_t>, !gccjit.array<!gccjit.int<size_t>, 3>, !gccjit.array<!gccjit.int<size_t>, 3>) -> !gccjit.struct<"memref<133x723x?xf32>" {#gccjit.field<"base" !gccjit.ptr<!gccjit.fp<float>>>, #gccjit.field<"aligned" !gccjit.ptr<!gccjit.fp<float>>>, #gccjit.field<"offset" !gccjit.int<size_t>>, #gccjit.field<"sizes" !gccjit.array<!gccjit.int<size_t>, 3>>, #gccjit.field<"strides" !gccjit.array<!gccjit.int<size_t>, 3>>}>
    // CHECK:   gccjit.return %[[V14]] : !gccjit.struct<"memref<133x723x?xf32>" {#gccjit.field<"base" !gccjit.ptr<!gccjit.fp<float>>>, #gccjit.field<"aligned" !gccjit.ptr<!gccjit.fp<float>>>, #gccjit.field<"offset" !gccjit.int<size_t>>, #gccjit.field<"sizes" !gccjit.array<!gccjit.int<size_t>, 3>>, #gccjit.field<"strides" !gccjit.array<!gccjit.int<size_t>, 3>>}>
    // CHECK: } : !gccjit.struct<"memref<133x723x?xf32>" {#gccjit.field<"base" !gccjit.ptr<!gccjit.fp<float>>>, #gccjit.field<"aligned" !gccjit.ptr<!gccjit.fp<float>>>, #gccjit.field<"offset" !gccjit.int<size_t>>, #gccjit.field<"sizes" !gccjit.array<!gccjit.int<size_t>, 3>>, #gccjit.field<"strides" !gccjit.array<!gccjit.int<size_t>, 3>>}>
    %a = memref.alloca (%arg0) : memref<133x723x?xf32>
    return
  }

  func.func @baz(%arg0 : index) {
    // CHECK: %[[V:[0-9]+]] = gccjit.const #gccjit.int<128> : !gccjit.int<size_t>
    // CHECK: gccjit.call  builtin @alloca_with_align(%{{[0-9]+}}, %[[V]]) : (!gccjit.int<size_t>, !gccjit.int<size_t>) -> !gccjit.ptr<!gccjit.void>
    %a = memref.alloca (%arg0) {alignment = 128} : memref<133x723x?xf32>
    return
  }
}
