// RUN: %gccjit-opt -o %t.mlir %s
// RUN: %filecheck --input-file=%t.mlir %s

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
    // CHECK-LABEL: @foo
    // CHECK-NEXT:     ^{{.+}}(%[[ARG0:.+]]: !gccjit.lvalue<!gccjit.int<bool>>):
    // CHECK-NEXT:         %[[#V0:]] = gccjit.as_rvalue %[[ARG0:.+]] : <!gccjit.int<bool>> to !gccjit.int<bool>
    // CHECK-NEXT:         gccjit.conditional(%[[#V0]] : <bool>), ^[[BB1:.+]], ^[[BB2:.+]]
    // CHECK-NEXT:     ^[[BB1]]:
    // CHECK-NEXT:         gccjit.return
    // CHECK-NEXT:     ^[[BB2]]:
    // CHECK-NEXT:         gccjit.return
    // CHECK-NEXT: }

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
    // CHECK-LABEL: @bar
    // CHECK-NEXT: ^{{.+}}(%[[ARG0:.+]]: !gccjit.lvalue<!gccjit.int<long>>):
    // CHECK-NEXT:     %[[#V0:]] = gccjit.as_rvalue %[[ARG0]] : <!gccjit.int<long>> to !gccjit.int<long>
    // CHECK-NEXT:     gccjit.switch(%[[#V0]] : <long>) {
    // CHECK-NEXT:         default -> ^[[BB3:.+]],
    // CHECK-NEXT:         #gccjit.int<5> : !gccjit.int<long> -> ^[[BB1:.+]],
    // CHECK-NEXT:         #gccjit.int<10> : !gccjit.int<long>...#gccjit.int<20> : !gccjit.int<long> -> ^[[BB2:.+]]
    // CHECK-NEXT:     }
    // CHECK-NEXT: ^[[BB1]]:
    // CHECK-NEXT:     gccjit.return
    // CHECK-NEXT: ^[[BB2]]:
    // CHECK-NEXT:     gccjit.return
    // CHECK-NEXT: ^[[BB3]]:
    // CHECK-NEXT:     gccjit.return
    // CHECK-NEXT: }
}
