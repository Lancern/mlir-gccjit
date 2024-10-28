// RUN: %gccjit-opt %s

!float = !gccjit.fp<float>
!ldb = !gccjit.fp<long double>
module @test {
    gccjit.func exported @foo(!float , !ldb) -> !float {
        ^entry(%arg0: !gccjit.lvalue<!float>, %arg1: !gccjit.lvalue<!ldb>):
            %0 = gccjit.zero : !float
            gccjit.return %0 : !float
    }
}
