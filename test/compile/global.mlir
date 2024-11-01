// RUN: %gccjit-translate -o %t.gimple %s -mlir-to-gccjit-gimple
// RUN: %filecheck --input-file=%t.gimple %s --check-prefix=CHECK-GIMPLE
// RUN: %gccjit-translate -o %t.exe %s -mlir-to-gccjit-executable && chmod +x %t.exe
// RUN: %t.exe | %filecheck %s --check-prefix=CHECK-OUTPUT
!i32 = !gccjit.int<int32_t>
!var32 = !gccjit.lvalue<!i32>
!char = !gccjit.int<char>
!const_char = !gccjit.qualified<!char, const>
!str = !gccjit.ptr<!const_char>
!i32_ptr = !gccjit.ptr<!i32>
!i32_arr10 = !gccjit.array<!i32, 10>
!void_ptr = !gccjit.ptr<!gccjit.void>
!varptr = !gccjit.lvalue<!i32_ptr>
!size_t = !gccjit.int<size_t>
!i1 = !gccjit.int<bool>
module @test attributes {
    gccjit.opt_level = #gccjit.opt_level<O3>,
    gccjit.prog_name = "test",
    gccjit.allow_unreachable = false,
    gccjit.debug_info = true
} {
    gccjit.func imported @printf(!str, ...) -> !i32

    gccjit.global internal @integer_array array(#gccjit.byte_array<[
        0,0,0,0,
        1,0,0,0,
        2,0,0,0,
        3,0,0,0,
        4,0,0,0,
        5,0,0,0,
        6,0,0,0,
        7,0,0,0,
        8,0,0,0,
        9,0,0,0
    ]>) : !gccjit.lvalue<!i32_arr10>

    gccjit.func internal @sum(!i32) -> !i32 {
        ^entry(%arg0: !var32):
            // here we use alloca on purpose to test the alloca lowering
            %0 = gccjit.as_rvalue %arg0 : !var32 to !i32
            %1 = gccjit.cast %0 : !i32 to !size_t
            %2 = gccjit.call builtin @__builtin_alloca(%1) : (!size_t) -> !void_ptr
            %3 = gccjit.bitcast %2 : !void_ptr to !i32_ptr
            %4 = gccjit.deref (%3 : !i32_ptr) : !var32 // acc
            %5 = gccjit.local : !var32 // ivar
            gccjit.jump ^loop_start

        ^loop_start:
            %6 = gccjit.as_rvalue %5 : !var32 to !i32
            %7 = gccjit.compare eq (%6 : !i32, %0 : !i32) : !i1
            gccjit.conditional (%7 : !i1), ^loop_end, ^loop_body

        ^loop_body:
            // CHECK-GIMPLE: %8 = (bitcast(&integer_array, __int32_t *))[%5];
            %8 = gccjit.expr {
              %10 = gccjit.as_rvalue %5 : !var32 to !i32
              %11 = gccjit.get_global @integer_array : !gccjit.lvalue<!i32_arr10>
              %12 = gccjit.addr (%11 : !gccjit.lvalue<!i32_arr10>) : !gccjit.ptr<!i32_arr10>
              %13 = gccjit.bitcast %12 : !gccjit.ptr<!i32_arr10> to !i32_ptr
              %14 = gccjit.array_access %13 [%10] : (!i32_ptr, !i32) -> !gccjit.lvalue<!i32>
              %15 = gccjit.as_rvalue %14 : !var32 to !i32
              gccjit.return %15 : !i32
            } : !i32
            // CHECK-GIMPLE: *%3 += %8;
            gccjit.update plus %8  to %4 : !i32, !var32
            %9 = gccjit.const #gccjit.one : !i32
            // CHECK-GIMPLE: %5 += %9;
            gccjit.update plus %9  to %5 : !i32, !var32
            gccjit.jump ^loop_start

        ^loop_end:
            %10 = gccjit.as_rvalue %4 : !var32 to !i32
            gccjit.return %10 : !i32
    }

    // CHECK-OUTPUT: sum = 45
    gccjit.func exported @main() -> !i32 {
        %0 = gccjit.const #gccjit.int<10> : !i32
        %1 = gccjit.call @sum(%0) : (!i32) -> !i32
        %2 = gccjit.literal "sum = %d\n" : !str
        %3 = gccjit.call @printf(%2, %1) : (!str, !i32) -> !i32
        %4 = gccjit.const #gccjit.zero : !i32
        gccjit.return %4 : !i32
    }

    
}
