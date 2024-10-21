module @test {
    gccjit.func exported @foo(i32, i32) -> i32 {
        ^entry(%arg0: !gccjit.lvalue<i32>, %arg1: !gccjit.lvalue<i32>):
            llvm.unreachable
    }
}
