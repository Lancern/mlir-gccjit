// RUN: %gccjit-opt -o %t.mlir %s
// RUN: %filecheck --input-file=%t.mlir %s

!i32 = !gccjit.int<int32_t>
module @test attributes {
    gccjit.opt_level = #gccjit.opt_level<O0>,
    gccjit.prog_name = "test",
    gccjit.allow_unreachable = false
} {
    gccjit.func exported @foo(!i32 , !i32) -> !i32 {
        ^entry(%arg0: !gccjit.lvalue<!i32>, %arg1: !gccjit.lvalue<!i32>):
            llvm.unreachable
    }
    // CHECK-LABEL: @foo
    // CHECK-NEXT:     ^{{.+}}(%{{.+}}: !gccjit.lvalue<!gccjit.int<int32_t>>, %{{.+}}: !gccjit.lvalue<!gccjit.int<int32_t>>):
    // CHECK-NEXT:         llvm.unreachable
    // CHECK-NEXT: }

    gccjit.func imported @bar(!gccjit.int<int>)
    // CHECK-LABEL: gccjit.func imported @bar (!gccjit.int<int>)
}
