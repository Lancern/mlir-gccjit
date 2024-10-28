// RUN: %gccjit-opt %s

!i32 = !gccjit.int<int32_t>
!ptr_i32 = !gccjit.ptr<!i32>
module @test attributes {
    gccjit.debug_info = true
} {
    gccjit.func exported @foo(!i32) -> !i32 attrs([
        #gccjit.fn_attr<noinline>
    ]) {
        ^entry(%arg0: !gccjit.lvalue<!i32>):
            %0 = gccjit.local align(16) : <!i32>
            %1 = gccjit.const #gccjit.one : !i32
            gccjit.assign %1 to %0 : !i32, <!i32>
            %2 = gccjit.as_rvalue %0 : !gccjit.lvalue<!i32> to !i32
            %3 = gccjit.as_rvalue %arg0 : !gccjit.lvalue<!i32> to !i32
            %4 = gccjit.binary plus (%2 : !i32, %3 : !i32) : !i32
            gccjit.return %4 : !i32
    }

    gccjit.global imported @test : !gccjit.lvalue<!i32>
    gccjit.global internal @test2 array(#gccjit.byte_array<[0, 0, 0, 0]>) : !gccjit.lvalue<!gccjit.array<!i32, 1>>
    gccjit.global exported @test3 init {
        %0 = gccjit.const #gccjit.zero : !i32
        gccjit.return %0 : !i32
    } : !gccjit.lvalue<!i32>
    gccjit.global exported @test4 literal (#gccjit.str<"hello, world!">) : !gccjit.lvalue<!gccjit.array<!gccjit.int<char>, 14>>
    gccjit.global exported @test5 link_section(#gccjit.link_section<".rodata">) init {
        %0 = gccjit.get_global @test3 : !gccjit.lvalue<!i32>
        %addr = gccjit.addr (%0 : !gccjit.lvalue<!i32>) : !gccjit.ptr<!i32>
        gccjit.return %addr : !gccjit.ptr<!i32>
    } : !gccjit.lvalue<!gccjit.ptr<!i32>>
}
