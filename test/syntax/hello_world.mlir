// RUN: %gccjit-opt -o %t.mlir %s
// RUN: %filecheck --input-file=%t.mlir %s

!i32 = !gccjit.int<int32_t>
!char = !gccjit.int<char>
!const_char = !gccjit.qualified<!char, const>
!str = !gccjit.ptr<!const_char>
module @test attributes {
    gccjit.opt_level = #gccjit.opt_level<O0>,
    gccjit.prog_name = "test",
    gccjit.allow_unreachable = false,
    gccjit.debug_info = true
} {
    gccjit.func imported @puts(!str) -> !i32
    // CHECK-LABEL: gccjit.func imported @puts (!gccjit.ptr<!gccjit.qualified<!gccjit.int<char>, const>>) -> !gccjit.int<int32_t>

    // CHECK-LABEL: @main
    gccjit.func exported @main() -> !i32 {
        %0 = gccjit.literal <"hello, world!\n"> : !str
        %1 = gccjit.call @puts(%0) : (!str) -> !i32
        %2 = gccjit.const #gccjit.zero : !i32
        gccjit.return %2 : !i32

        // CHECK-NEXT: %[[#V0:]] = gccjit.literal <"hello, world!\0A"> : <!gccjit.qualified<!gccjit.int<char>, const>>
        // CHECK-NEXT: %[[#V1:]] = gccjit.call @puts(%[[#V0]]) : (!gccjit.ptr<!gccjit.qualified<!gccjit.int<char>, const>>) -> !gccjit.int<int32_t>
        // CHECK-NEXT: %[[#V2:]] = gccjit.const #gccjit.zero : !gccjit.int<int32_t>
        // CHECK-NEXT: gccjit.return %[[#V2]] : !gccjit.int<int32_t>
    }
}
