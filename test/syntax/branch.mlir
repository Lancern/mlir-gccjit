// RUN: %gccjit-opt %s

!bool = !gccjit.int<bool>
!long = !gccjit.int<long>
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
    gccjit.func exported @bar(!long) {
        ^entry(%arg0: !gccjit.lvalue<!long>):
            %0 = gccjit.as_rvalue %arg0 : !gccjit.lvalue<!long> to !long
            gccjit.switch (%0 : !long) {
                default -> ^default,
                #gccjit.int<5> : !long -> ^case1,
                #gccjit.int<10> : !long ...#gccjit.int<20> : !long -> ^case2
            }
        ^case1:
            gccjit.return
        ^case2:
            gccjit.return
        ^default:
            gccjit.return
    }
}
