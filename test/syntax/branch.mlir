!bool = !gccjit.int<bool>
module @test {
    gccjit.func exported @foo(!bool) {
        ^entry(%arg0: !gccjit.lvalue<!bool>):
            %0 = gccjit.as_rvalue %arg0 : !gccjit.lvalue<!bool> to !bool
            gccjit.conditional (%0 : !bool), ^true, ^false
        ^true:
            gccjit.return
        ^false:
            gccjit.return
    }
}
