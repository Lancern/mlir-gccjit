// RUN: %gccjit-translate -o %t.gimple %s -mlir-to-gccjit-gimple
// RUN: %filecheck --input-file=%t.gimple %s --check-prefix=CHECK-GIMPLE
!i32 = !gccjit.int<int32_t>
!var32 = !gccjit.lvalue<!i32>
!bool = !gccjit.int<bool>
!varbool = !gccjit.lvalue<!bool>
!float = !gccjit.fp<float>
!varfloat = !gccjit.lvalue<!float>
module @unary_testing {
    gccjit.func exported @unary_minus(!i32) -> !i32 {
        ^body(%arg0: !var32):
            // CHECK-GIMPLE:  %[[LOAD:[0-9]+]] = %arg0;
            %0 = gccjit.as_rvalue %arg0 : !var32 to !i32
            // CHECK-GIMPLE:  %[[RET:[0-9]+]] = -(%[[LOAD]]);
            %1 = gccjit.unary minus (%0 : !i32) : !i32
            // CHECK-GIMPLE: return %[[RET]];
            gccjit.return %1 : !i32
    }
    gccjit.func exported @unary_minus_lv(!i32) -> !i32 {
        ^body(%arg0: !var32):
            // CHECK-GIMPLE:  %[[RET:[0-9]+]] = -(%arg0);
            // CHECK-GIMPLE: return %[[RET]];
            %0 = gccjit.unary minus (%arg0 : !var32) : !i32
            gccjit.return %0 : !i32
    }
    gccjit.func exported @unary_bitwise_negate(!i32) -> !i32 {
        ^body(%arg0: !var32):
            // CHECK: return ~(%arg0);
            %0 = gccjit.expr lazy {
                %1 = gccjit.unary bitwise_negate (%arg0 : !var32) : !i32
                gccjit.return %1 : !i32
            } : !i32
            gccjit.return %0 : !i32
    }
    gccjit.func exported @unary_logical_negate(!bool) -> !bool {
        ^body(%arg0: !varbool):
            // CHECK: return !(%arg0);
            %0 = gccjit.expr lazy {
                %1 = gccjit.unary logical_negate (%arg0 : !varbool) : !bool
                gccjit.return %1 : !bool
            } : !bool
            gccjit.return %0 : !bool
    }
    gccjit.func exported @unary_abs(!float) -> !float {
        ^body(%arg0: !varfloat):
            // CHECK: return abs (%arg0);
            %0 = gccjit.expr lazy {
                %1 = gccjit.unary abs (%arg0 : !varfloat) : !float
                gccjit.return %1 : !float
            } : !float
            gccjit.return %0 : !float
    }
}
