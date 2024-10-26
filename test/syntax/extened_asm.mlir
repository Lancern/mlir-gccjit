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
}
