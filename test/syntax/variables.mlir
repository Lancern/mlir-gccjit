// RUN: %gccjit-opt -o %t.mlir %s
// RUN: %filecheck --input-file=%t.mlir %s

!i32 = !gccjit.int<int32_t>
!ptr_i32 = !gccjit.ptr<!i32>
module @test attributes {
    gccjit.debug_info = true
} {
    // CHECK-LABEL: @foo
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

        //      CHECK: ^{{.+}}(%arg0: !gccjit.lvalue<!gccjit.int<int32_t>>):
        // CHECK-NEXT:     %[[#V0:]] = gccjit.local align(16) : <!gccjit.int<int32_t>>
        // CHECK-NEXT:     %[[#V1:]] = gccjit.const #gccjit.one : !gccjit.int<int32_t>
        // CHECK-NEXT:     gccjit.assign %[[#V1]] to %[[#V0]] : !gccjit.int<int32_t>, <!gccjit.int<int32_t>>
        // CHECK-NEXT:     %[[#V2:]] = gccjit.as_rvalue %[[#V0]] : <!gccjit.int<int32_t>> to !gccjit.int<int32_t>
        // CHECK-NEXT:     %[[#V3:]] = gccjit.as_rvalue %arg0 : <!gccjit.int<int32_t>> to !gccjit.int<int32_t>
        // CHECK-NEXT:     %[[#V4:]] = gccjit.binary plus(%[[#V2]] : !gccjit.int<int32_t>, %[[#V3]] : !gccjit.int<int32_t>) : !gccjit.int<int32_t>
        // CHECK-NEXT:     gccjit.return %[[#V4]] : !gccjit.int<int32_t>
    }

    gccjit.global imported @test : !gccjit.lvalue<!i32>
    // CHECK-LABEL: gccjit.global imported @test  : !gccjit.lvalue<!gccjit.int<int32_t>>

    gccjit.global internal @test2 array(#gccjit.byte_array<[0, 0, 0, 0]>) : !gccjit.lvalue<!gccjit.array<!i32, 1>>
    // CHECK-LABEL: gccjit.global internal @test2 array(<[0, 0, 0, 0]>) : !gccjit.lvalue<!gccjit.array<!gccjit.int<int32_t>, 1>>

    // CHECK-LABEL: @test3
    gccjit.global exported @test3 init {
        %0 = gccjit.const #gccjit.zero : !i32
        gccjit.return %0 : !i32

        //      CHECK: %[[#V0:]] = gccjit.const #gccjit.zero : !gccjit.int<int32_t>
        // CHECK-NEXT: gccjit.return %[[#V0]] : !gccjit.int<int32_t>
    } : !gccjit.lvalue<!i32>

    gccjit.global exported @test4 literal ("hello, world!") : !gccjit.lvalue<!gccjit.array<!gccjit.int<char>, 14>>
    // CHECK-LABEL: gccjit.global exported @test4 literal("hello, world!") : !gccjit.lvalue<!gccjit.array<!gccjit.int<char>, 14>>

    // CHECK-LABEL: @test5
    gccjit.global exported @test5 link_section(".rodata") init {
        %0 = gccjit.get_global @test3 : !gccjit.lvalue<!i32>
        %addr = gccjit.addr (%0 : !gccjit.lvalue<!i32>) : !gccjit.ptr<!i32>
        gccjit.return %addr : !gccjit.ptr<!i32>

        //      CHECK: %[[#V0:]] = gccjit.get_global @test3 : <!gccjit.int<int32_t>>
        // CHECK-NEXT: %[[#V1:]] = gccjit.addr(%[[#V0]] : <!gccjit.int<int32_t>>) : <!gccjit.int<int32_t>>
        // CHECK-NEXT: gccjit.return %[[#V1]] : !gccjit.ptr<!gccjit.int<int32_t>>
    } : !gccjit.lvalue<!gccjit.ptr<!i32>>
}
