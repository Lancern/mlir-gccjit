// RUN: %gccjit-translate -o %t.gimple %s -mlir-to-gccjit-gimple
// RUN: %filecheck --input-file=%t.gimple %s --check-prefix=CHECK-GIMPLE

!i32 = !gccjit.int<int32_t>
!i1 = !gccjit.int<bool>
!var32 = !gccjit.lvalue<!i32>
!char = !gccjit.int<char>
!const_char = !gccjit.qualified<!char, const>
!str = !gccjit.ptr<!const_char>
module @test attributes {
    gccjit.opt_level = #gccjit.opt_level<O3>,
    gccjit.prog_name = "test",
    gccjit.allow_unreachable = false,
    gccjit.debug_info = true
} {
    // fuse expr into return
    gccjit.func exported @add(!i32, !i32) -> !i32 {
        ^body(%arg0: !var32, %arg1: !var32):
            %res = gccjit.expr lazy {
                %0 = gccjit.as_rvalue %arg0 : !var32 to !i32
                %1 = gccjit.as_rvalue %arg1 : !var32 to !i32
                %2 = gccjit.binary plus (%0 : !i32, %1 : !i32) : !i32
                gccjit.return %2 : !i32
            } : !i32
            // CHECK-GIMPLE: return %arg0 + %arg1;
            gccjit.return %res : !i32
    }
    
    // fuse expr into conditional
    gccjit.func exported @max(!i32, !i32) -> !i32 {
        ^body(%arg0: !var32, %arg1: !var32):
            %0 = gccjit.as_rvalue %arg0 : !var32 to !i32
            %1 = gccjit.as_rvalue %arg1 : !var32 to !i32
            %3 = gccjit.expr lazy {
                %2 = gccjit.compare gt (%0 : !i32, %1 : !i32) : !i1
                gccjit.return %2 : !i1
            } : !i1
            // CHECK-GIMPLE: if (%0 > %1) goto bb1; else goto bb2;
            gccjit.conditional (%3 : !i1), ^bb1, ^bb2
        ^bb1:
            gccjit.return %0 : !i32
        ^bb2:
            gccjit.return %1 : !i32
    }

    
}
