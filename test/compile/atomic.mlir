// RUN: %gccjit-translate -o %t.gimple %s -mlir-to-gccjit-gimple
// RUN: %filecheck --input-file=%t.gimple %s

!bool = !gccjit.int<bool>
!i32 = !gccjit.int<int32_t>
!f32 = !gccjit.fp<float>
!pi32 = !gccjit.ptr<!i32>
!pf32 = !gccjit.ptr<!f32>
!ppi32 = !gccjit.ptr<!pi32>

module @test attributes {
    gccjit.opt_level = #gccjit.opt_level<O0>,
    gccjit.prog_name = "test",
    gccjit.allow_unreachable = false,
    gccjit.debug_info = true
} {
    // CHECK-LABEL: atomic_load_int
    gccjit.func exported @atomic_load_int(!pi32) -> !i32 {
        ^entry(%arg : !gccjit.lvalue<!pi32>):
            %0 = gccjit.as_rvalue %arg : !gccjit.lvalue<!pi32> to !pi32
            // CHECK: %{{.+}} = __atomic_load_4 (((volatile const void *)%{{.+}}), (int)0);
            %1 = gccjit.atomic.load relaxed (%0 : !pi32) : !i32
            gccjit.return %1 : !i32
    }

    // CHECK-LABEL: atomic_load_float
    gccjit.func exported @atomic_load_float(!pf32) -> !f32 {
        ^entry(%arg : !gccjit.lvalue<!pf32>):
            %0 = gccjit.as_rvalue %arg : !gccjit.lvalue<!pf32> to !pf32
            // CHECK: %{{.+}} = bitcast(__atomic_load_4 (((volatile const void *)%{{.+}}), (int)0), float);
            %1 = gccjit.atomic.load relaxed (%0 : !pf32) : !f32
            gccjit.return %1 : !f32
    }

    // CHECK-LABEL: atomic_load_ptr
    gccjit.func exported @atomic_load_ptr(!ppi32) -> !pi32 {
        ^entry(%arg : !gccjit.lvalue<!ppi32>):
            %0 = gccjit.as_rvalue %arg : !gccjit.lvalue<!ppi32> to !ppi32
            // CHECK: %{{.+}} = bitcast(__atomic_load_8 (((volatile const void *)%{{.+}}), (int)0), __int32_t *);
            %1 = gccjit.atomic.load relaxed (%0 : !ppi32) : !pi32
            gccjit.return %1 : !pi32
    }

    // CHECK-LABEL: atomic_store_int
    gccjit.func exported @atomic_store_int(!pi32, !i32) {
        ^entry(%arg0 : !gccjit.lvalue<!pi32>, %arg1 : !gccjit.lvalue<!i32>):
            %0 = gccjit.as_rvalue %arg0 : !gccjit.lvalue<!pi32> to !pi32
            %1 = gccjit.as_rvalue %arg1 : !gccjit.lvalue<!i32> to !i32
            // CHECK: (void)__atomic_store_4 (((volatile void *)%{{.+}}), %{{.+}}, (int)0);
            gccjit.atomic.store relaxed (%0 : !pi32, %1 : !i32)
            gccjit.return
    }

    // CHECK-LABEL: atomic_store_float
    gccjit.func exported @atomic_store_float(!pf32, !f32) {
        ^entry(%arg0 : !gccjit.lvalue<!pf32>, %arg1 : !gccjit.lvalue<!f32>):
            %0 = gccjit.as_rvalue %arg0 : !gccjit.lvalue<!pf32> to !pf32
            %1 = gccjit.as_rvalue %arg1 : !gccjit.lvalue<!f32> to !f32
            // CHECK: (void)__atomic_store_4 (((volatile void *)%{{.+}}), (bitcast(%{{.+}}, int)), (int)0);
            gccjit.atomic.store relaxed (%0 : !pf32, %1 : !f32)
            gccjit.return
    }

    // CHECK-LABEL: atomic_store_ptr
    gccjit.func exported @atomic_store_ptr(!ppi32, !pi32) {
        ^entry(%arg0 : !gccjit.lvalue<!ppi32>, %arg1 : !gccjit.lvalue<!pi32>):
            %0 = gccjit.as_rvalue %arg0 : !gccjit.lvalue<!ppi32> to !ppi32
            %1 = gccjit.as_rvalue %arg1 : !gccjit.lvalue<!pi32> to !pi32
            // CHECK: (void)__atomic_store_8 (((volatile void *)%{{.+}}), (bitcast(%{{.+}}, long long)), (int)0);
            gccjit.atomic.store relaxed (%0 : !ppi32, %1 : !pi32)
            gccjit.return
    }

    // CHECK-LABEL: atomic_rmw_int
    gccjit.func exported @atomic_rmw_int(!pi32, !i32) -> !i32 {
        ^entry(%arg0 : !gccjit.lvalue<!pi32>, %arg1 : !gccjit.lvalue<!i32>):
            %0 = gccjit.as_rvalue %arg0 : !gccjit.lvalue<!pi32> to !pi32
            %1 = gccjit.as_rvalue %arg1 : !gccjit.lvalue<!i32> to !i32
            // CHECK: %{{.+}} = __atomic_fetch_add_4 (((volatile void *)%{{.+}}), %{{.+}}, (int)0);
            %2 = gccjit.atomic.rmw relaxed fetch_add (%0 : !pi32, %1 : !i32) : !i32
            gccjit.return %2 : !i32
    }

    // CHECK-LABEL: atomic_rmw_float
    gccjit.func exported @atomic_rmw_float(!pf32, !f32) -> !f32 {
        ^entry(%arg0 : !gccjit.lvalue<!pf32>, %arg1 : !gccjit.lvalue<!f32>):
            %0 = gccjit.as_rvalue %arg0 : !gccjit.lvalue<!pf32> to !pf32
            %1 = gccjit.as_rvalue %arg1 : !gccjit.lvalue<!f32> to !f32
            // CHECK: %{{.+}} = bitcast(__atomic_fetch_add_4 (((volatile void *)%{{.+}}), (bitcast(%{{.+}}, int)), (int)0), float);
            %2 = gccjit.atomic.rmw relaxed fetch_add (%0 : !pf32, %1 : !f32) : !f32
            gccjit.return %2 : !f32
    }

    // CHECK-LABEL: atomic_rmw_ptr
    gccjit.func exported @atomic_rmw_ptr(!ppi32, !pi32) -> !pi32 {
        ^entry(%arg0 : !gccjit.lvalue<!ppi32>, %arg1 : !gccjit.lvalue<!pi32>):
            %0 = gccjit.as_rvalue %arg0 : !gccjit.lvalue<!ppi32> to !ppi32
            %1 = gccjit.as_rvalue %arg1 : !gccjit.lvalue<!pi32> to !pi32
            // CHECK: %{{.+}} = bitcast(__atomic_fetch_add_8 (((volatile void *)%{{.+}}), (bitcast(%{{.+}}, long long)), (int)0), __int32_t *);
            %2 = gccjit.atomic.rmw relaxed fetch_add (%0 : !ppi32, %1 : !pi32) : !pi32
            gccjit.return %2 : !pi32
    }

    // CHECK-LABEL: atomic_cmpxchg_int
    gccjit.func exported @atomic_cmpxchg_int(!pi32, !pi32, !i32) -> !bool {
        ^entry(%arg0 : !gccjit.lvalue<!pi32>, %arg1 : !gccjit.lvalue<!pi32>, %arg2 : !gccjit.lvalue<!i32>):
            %0 = gccjit.as_rvalue %arg0 : !gccjit.lvalue<!pi32> to !pi32
            %1 = gccjit.as_rvalue %arg1 : !gccjit.lvalue<!pi32> to !pi32
            %2 = gccjit.as_rvalue %arg2 : !gccjit.lvalue<!i32> to !i32
            // CHECK: %{{.+}} = __atomic_compare_exchange_4 (((volatile void *)%{{.+}}), ((volatile const void *)%{{.+}}), %{{.+}}, (bool)1, (int)4, (int)0);
            %3 = gccjit.atomic.cmpxchg weak success(acq_rel) failure(relaxed) (%0 : !pi32, %1 : !pi32, %2 : !i32) : !bool
            gccjit.return %3 : !bool
    }

    // CHECK-LABEL: atomic_cmpxchg_float
    gccjit.func exported @atomic_cmpxchg_float(!pf32, !pf32, !f32) -> !bool {
        ^entry(%arg0 : !gccjit.lvalue<!pf32>, %arg1 : !gccjit.lvalue<!pf32>, %arg2 : !gccjit.lvalue<!f32>):
            %0 = gccjit.as_rvalue %arg0 : !gccjit.lvalue<!pf32> to !pf32
            %1 = gccjit.as_rvalue %arg1 : !gccjit.lvalue<!pf32> to !pf32
            %2 = gccjit.as_rvalue %arg2 : !gccjit.lvalue<!f32> to !f32
            // CHECK: %{{.+}} = __atomic_compare_exchange_4 (((volatile void *)%{{.+}}), ((volatile const void *)%{{.+}}), (bitcast(%{{.+}}, int)), (bool)1, (int)4, (int)0);
            %3 = gccjit.atomic.cmpxchg weak success(acq_rel) failure(relaxed) (%0 : !pf32, %1 : !pf32, %2 : !f32) : !bool
            gccjit.return %3 : !bool
    }

    // CHECK-LABEL: atomic_cmpxchg_ptr
    gccjit.func exported @atomic_cmpxchg_ptr(!ppi32, !ppi32, !pi32) -> !bool {
        ^entry(%arg0 : !gccjit.lvalue<!ppi32>, %arg1 : !gccjit.lvalue<!ppi32>, %arg2 : !gccjit.lvalue<!pi32>):
            %0 = gccjit.as_rvalue %arg0 : !gccjit.lvalue<!ppi32> to !ppi32
            %1 = gccjit.as_rvalue %arg1 : !gccjit.lvalue<!ppi32> to !ppi32
            %2 = gccjit.as_rvalue %arg2 : !gccjit.lvalue<!pi32> to !pi32
            // CHECK: %{{.+}} = __atomic_compare_exchange_8 (((volatile void *)%{{.+}}), ((volatile const void *)%{{.+}}), (bitcast(%{{.+}}, long long)), (bool)1, (int)4, (int)0);
            %3 = gccjit.atomic.cmpxchg weak success(acq_rel) failure(relaxed) (%0 : !ppi32, %1 : !ppi32, %2 : !pi32) : !bool
            gccjit.return %3 : !bool
    }
}
