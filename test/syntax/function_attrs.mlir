// RUN: %gccjit-opt %s

!i32 = !gccjit.int<int32_t>
!ptr_i32 = !gccjit.ptr<!i32>
module @test attributes {
    gccjit.opt_level = #gccjit.opt_level<O0>,
    gccjit.prog_name = "test",
    gccjit.allow_unreachable = false
} {
    gccjit.func exported @foo(!ptr_i32) -> !ptr_i32 attrs([
        #gccjit.fn_attr<inline>,
        #gccjit.fn_attr<target, "znver4">,
        #gccjit.fn_attr<nonnull, array<i32: 0>>])
    {
        ^entry(%arg0: !gccjit.lvalue<!ptr_i32>):
            %0 = gccjit.as_rvalue %arg0 : !gccjit.lvalue<!ptr_i32> to !ptr_i32
            gccjit.return %0 : !ptr_i32
    }

}
