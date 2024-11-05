// RUN: %gccjit-opt -o %t.mlir %s
// RUN: %filecheck --input-file=%t.mlir %s

module @test {
    gccjit.func imported @gemm (
        !gccjit.struct<"__memref_188510220862752" {
            #gccjit.field<"base" !gccjit.ptr<!gccjit.fp<float>>>,
            #gccjit.field<"aligned" !gccjit.ptr<!gccjit.fp<float>>>,
            #gccjit.field<"offset" !gccjit.int<size_t> : 32>,
            #gccjit.field<"sizes" !gccjit.array<!gccjit.int<size_t>, 2>>,
            #gccjit.field<"strides" !gccjit.array<!gccjit.int<size_t>, 2>>
        }>
    )
    // CHECK: @gemm
    // CHECK-SAME: !gccjit.struct<"__memref_188510220862752" {
    // CHECK-SAME:   #gccjit.field<"base" !gccjit.ptr<!gccjit.fp<float>>>
    // CHECK-SAME:   #gccjit.field<"aligned" !gccjit.ptr<!gccjit.fp<float>>>
    // CHECK-SAME:   #gccjit.field<"offset" !gccjit.int<size_t> : 32>
    // CHECK-SAME:   #gccjit.field<"sizes" !gccjit.array<!gccjit.int<size_t>, 2>>
    // CHECK-SAME:   #gccjit.field<"strides" !gccjit.array<!gccjit.int<size_t>, 2>>
    // CHECK-SAME: }
}
