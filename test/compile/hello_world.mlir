// RUN: %gccjit-translate -o %t.gimple %s -mlir-to-gccjit-gimple
// RUN: %filecheck --input-file=%t.gimple %s --check-prefix=CHECK-GIMPLE
// RUN: %gccjit-translate -o %t.exe %s -mlir-to-gccjit-executable && chmod +x %t.exe
// RUN: %t.exe | %filecheck %s --check-prefix=CHECK-OUTPUT

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

    gccjit.func exported @main() -> !i32 {
        %0 = gccjit.literal <"hello, world!\n"> : !str
        // CHECK-GIMPLE: (void)puts ("hello, world!\n");
        // CHECK-OUTPUT: hello, world!
        %1 = gccjit.call @puts(%0) : (!str) -> !i32 {gccjit.eval}
        %2 = gccjit.const #gccjit.zero : !i32
        // CHECK-GIMPLE: return (__int32_t)0;
        gccjit.return %2 : !i32
    }
}
