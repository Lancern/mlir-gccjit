// RUN: %gccjit-translate -o %t.gimple %s -mlir-to-gccjit-gimple
// RUN: %filecheck --input-file=%t.gimple %s --check-prefix=CHECK-GIMPLE
// RUN: %gccjit-translate -o %t.exe %s -mlir-to-gccjit-executable && chmod +x %t.exe
// RUN: echo 10 | %t.exe | %filecheck %s --check-prefix=CHECK-OUTPUT

!i32 = !gccjit.int<int32_t>
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
    gccjit.func imported @printf(!str, ...) -> !i32
    gccjit.func imported @scanf(!str, ...) -> !i32

    gccjit.func internal @fib(!i32 ) -> !i32 {
        ^entry(%arg : !var32):
            %0 = gccjit.as_rvalue %arg : !var32 to !i32
            gccjit.switch (%0 : !i32) {
                default -> ^bb1,
                #gccjit.int<0> : !i32 ... #gccjit.int<1> : !i32 -> ^bb0
            }
        ^bb0:
            %1 = gccjit.const #gccjit.one : !i32
            gccjit.return %1 : !i32
        ^bb1:
            // CHECK-GIMPLE: %2 = fib ((%0 - (__int32_t)1)) + fib ((%0 - (__int32_t)2));
            %9 = gccjit.expr {
                %2 = gccjit.const #gccjit.one : !i32
                %3 = gccjit.const #gccjit.int<2> : !i32
                %4 = gccjit.binary minus (%0 : !i32, %2 : !i32) : !i32
                %5 = gccjit.binary minus (%0 : !i32, %3 : !i32) : !i32
                %6 = gccjit.call @fib(%4) : (!i32) -> !i32
                %7 = gccjit.call @fib(%5) : (!i32) -> !i32
                %8 = gccjit.binary plus (%6 : !i32, %7 : !i32) : !i32
                gccjit.return %8 : !i32
            } : !i32
            gccjit.return %9 : !i32
    }

    // CHECK-OUTPUT: fib(10) = 89
    gccjit.func exported @main() -> !i32 {
        %var = gccjit.local : !var32
        %addr = gccjit.addr (%var : !var32) : !gccjit.ptr<!i32> 
        %scanformat = gccjit.literal "%d" : !str
        %discard = gccjit.call @scanf(%scanformat, %addr) : (!str, !gccjit.ptr<!i32>) -> !i32
        %val = gccjit.as_rvalue %var : !var32 to !i32
        %1 = gccjit.call @fib(%val) : (!i32) -> !i32
        %2 = gccjit.literal "fib(%d) = %d\n" : !str 
        // CHECK-GIMPLE: %{{[0-9]}} = printf (%{{[0-9]}}, %{{[0-9]}}, %{{[0-9]}});
        %3 = gccjit.call @printf(%2, %val, %1) : (!str, !i32, !i32) -> !i32
        %4 = gccjit.const #gccjit.zero : !i32
        gccjit.return %4 : !i32
    }

    
}
