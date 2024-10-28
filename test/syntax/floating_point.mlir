// RUN: %gccjit-opt -o %t.mlir %s
// RUN: %filecheck --input-file=%t.mlir %s

!float = !gccjit.fp<float>
!ldb = !gccjit.fp<long double>
module @test {
    gccjit.func exported @foo(!float , !ldb) -> !float {
        ^entry(%arg0: !gccjit.lvalue<!float>, %arg1: !gccjit.lvalue<!ldb>):
            %0 = gccjit.const #gccjit.zero : !float
            gccjit.return %0 : !float
    }
    // CHECK-LABEL: @foo
    // CHECK-NEXT:     ^{{.+}}(%{{.+}}: !gccjit.lvalue<!gccjit.fp<float>>, %{{.+}}: !gccjit.lvalue<!gccjit.fp<long double>>):
    // CHECK-NEXT:     %[[#V0:]] = gccjit.const #gccjit.zero : !gccjit.fp<float>
    // CHECK-NEXT:     gccjit.return %[[#V0]] : !gccjit.fp<float>
    // CHECK-NEXT: }
}
