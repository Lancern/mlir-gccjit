// RUN: %gccjit-opt -o %t.mlir %s
// RUN: %filecheck --input-file=%t.mlir %s

!u64 = !gccjit.int<uint64_t>
module @test {
    gccjit.func exported @foo() -> !u64 {
        %msr = gccjit.local : !gccjit.lvalue<!u64>
        gccjit.asm volatile (
            "rdtsc\n\tshl $32, %%rdx\n\tor %%rdx, %0\n\t"
            : "=a" (%msr : !gccjit.lvalue<!u64>)
            :
            : "rdx"
        )
        %val = gccjit.as_rvalue %msr : !gccjit.lvalue<!u64> to !u64
        gccjit.return %val : !u64
    }
    // CHECK-LABEL: @foo
    // CHECK-NEXT:     %[[#V0:]] = gccjit.local : <!gccjit.int<uint64_t>>
    // CHECK-NEXT:     gccjit.asm  volatile("rdtsc\0A\09shl $32, %%rdx\0A\09or %%rdx, %0\0A\09" : "=a"(%[[#V0]] : !gccjit.lvalue<!gccjit.int<uint64_t>>) :  : "rdx")
    // CHECK-NEXT:     %[[#V1:]] = gccjit.as_rvalue %[[#V0]] : <!gccjit.int<uint64_t>> to !gccjit.int<uint64_t>
    // CHECK-NEXT:     gccjit.return %[[#V1]] : !gccjit.int<uint64_t>
    // CHECK-NEXT: }
}
