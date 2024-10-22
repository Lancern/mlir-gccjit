!i32 = !gccjit.int<int32_t>
!ptr_i32 = !gccjit.ptr<!i32>
module @test attributes { 
    gccjit.debug_info = true
} {
    gccjit.func exported @foo(!i32) -> !i32  attrs([
        #gccjit.fn_attr<noinline> 
    ]) {
        ^entry(%arg0: !gccjit.lvalue<!i32>):
            %0 = gccjit.local align(16) : <!i32>
            %1 = gccjit.as_rvalue %0 : !gccjit.lvalue<!i32> to !i32
            gccjit.return %1 : !i32
    }

    gccjit.global imported @test link_section(".rodata") : !gccjit.lvalue<!i32>
    gccjit.global internal @test2 array([0, 0, 0, 0]) : !gccjit.lvalue<!gccjit.array<!i32, 1>>
    gccjit.global exported @test3 init {
        %0 = gccjit.const #gccjit.zero : !i32
        gccjit.return %0 : !i32
    } : !gccjit.lvalue<!i32>
    gccjit.global exported @test4 literal ("hello, world!") : !gccjit.lvalue<!gccjit.array<!gccjit.int<char>, 14>>
}
