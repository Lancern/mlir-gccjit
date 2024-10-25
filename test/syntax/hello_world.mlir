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
        %1 = gccjit.literal <"hello, world!\n"> : !str
        %2 = gccjit.call @puts(%1) : (!str) -> !i32 { gccjit.eval }
        %0 = gccjit.const #gccjit.zero : !i32
        gccjit.return %0 : !i32
    }
}
