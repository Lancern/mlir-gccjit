// RUN: %gccjit-opt -o %t.mlir %s
// RUN: %filecheck --input-file=%t.mlir %s

module @test {
    gccjit.func imported @gemm (
        !gccjit.struct<"__memref_188510220862752" {
            #gccjit.field<!gccjit.ptr<!gccjit.fp<float>> "base">,
            #gccjit.field<!gccjit.ptr<!gccjit.fp<float>> "aligned">,
            #gccjit.field<!gccjit.int<size_t> "offset">,
            #gccjit.field<!gccjit.array<!gccjit.int<size_t>, 2> "sizes">,
            #gccjit.field<!gccjit.array<!gccjit.int<size_t>, 2> "strides">
        }>
    )
    // CHECK: @gemm
    // CHECK-SAME: !gccjit.struct<"__memref_188510220862752" {
    // CHECK-SAME:   #gccjit.field<!gccjit.ptr<!gccjit.fp<float>> "base">
    // CHECK-SAME:   #gccjit.field<!gccjit.ptr<!gccjit.fp<float>> "aligned">
    // CHECK-SAME:   #gccjit.field<!gccjit.int<size_t> "offset">
    // CHECK-SAME:   #gccjit.field<!gccjit.array<!gccjit.int<size_t>, 2> "sizes">
    // CHECK-SAME:   #gccjit.field<!gccjit.array<!gccjit.int<size_t>, 2> "strides">
    // CHECK-SAME: }
}
