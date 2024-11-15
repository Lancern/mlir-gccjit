// RUN: %gccjit-translate -o %t.gimple %s -mlir-to-gccjit-gimple
// RUN: %filecheck --input-file=%t.gimple %s
// RUN: %gccjit-translate -o %t.exe %s -mlir-to-gccjit-executable && chmod +x %t.exe && %t.exe

!int = !gccjit.int<int>
!ilv = !gccjit.lvalue<!int>
!bool = !gccjit.int<bool>
!bitfields = !gccjit.struct<"Int" {
    #gccjit.field<"A" !int : 17>,
    #gccjit.field<"B" !int : 5>,
    #gccjit.field<"C" !int : 10>
}>


module attributes { gccjit.opt_level = #gccjit.opt_level<O3>, gccjit.allow_unreachable = true } {
    gccjit.func exported @from_int_to_bitfield (!int, !int, !int, !int) {
        ^entry(%arg0: !ilv, %arg1: !ilv, %arg2: !ilv, %arg3: !ilv):
            %0 = gccjit.as_rvalue %arg0 : !ilv to !int
            %1 = gccjit.as_rvalue %arg1 : !ilv to !int
            %2 = gccjit.as_rvalue %arg2 : !ilv to !int
            %3 = gccjit.as_rvalue %arg3 : !ilv to !int
            // CHECK: %[[V:[0-9]+]] = bitcast(%{{[0-9]+}}, struct Int);
            // CHECK: %{{[0-9]+}} = %[[V]].A:17;
            // CHECK: %{{[0-9]+}} = %[[V]].B:5;
            // CHECK: %{{[0-9]+}} = %[[V]].C:10;
            %4 = gccjit.bitcast %0 : !int to !bitfields
            %5 = gccjit.access_field %4[0] : !bitfields -> !int
            %6 = gccjit.access_field %4[1] : !bitfields -> !int
            %7 = gccjit.access_field %4[2] : !bitfields -> !int
            %eq0 = gccjit.compare eq (%5 : !int, %1 : !int) : !bool
            %eq1 = gccjit.compare eq (%6 : !int, %2 : !int) : !bool
            %eq2 = gccjit.compare eq (%7 : !int, %3 : !int) : !bool
            %and0 = gccjit.binary logical_and (%eq0 : !bool, %eq1 : !bool) : !bool
            %and1 = gccjit.binary logical_and (%and0 : !bool, %eq2 : !bool) : !bool
            gccjit.conditional (%and1 : !bool), ^return, ^trap

        ^return:
            gccjit.return
        
        ^trap:
            gccjit.call builtin @__builtin_trap() : () -> !gccjit.void
            gccjit.jump ^trap
    }
    gccjit.func exported @func_bitfield_to_int(!int, !int, !int, !int) {
        ^entry(%arg0: !ilv, %arg1: !ilv, %arg2: !ilv, %arg3: !ilv):
            %0 = gccjit.as_rvalue %arg0 : !ilv to !int
            %1 = gccjit.as_rvalue %arg1 : !ilv to !int
            %2 = gccjit.as_rvalue %arg2 : !ilv to !int
            %3 = gccjit.as_rvalue %arg3 : !ilv to !int
            // CHECK: (struct Int) {.A:17=%{{[0-9]+}}, .B:5=%{{[0-9]+}}, .C:10=%{{[0-9]+}}};
            %4 = gccjit.new_struct [0, 1, 2] [%0, %1, %2] : (!int, !int, !int) -> !bitfields
            %5 = gccjit.bitcast %4 : !bitfields to !int
            %eq = gccjit.compare eq (%5 : !int, %3 : !int) : !bool
            gccjit.conditional (%eq : !bool), ^return, ^trap
        
        ^return:
            gccjit.return

        ^trap:
            gccjit.call builtin @__builtin_trap() : () -> !gccjit.void
            gccjit.jump ^trap
    }

    gccjit.func exported @main() -> !int {
        ^entry:
            %0 = gccjit.const #gccjit.int<-559038737> : !int
            %1 = gccjit.const #gccjit.int<-16657> : !int
            %2 = gccjit.const #gccjit.int<-10> : !int
            %3 = gccjit.const #gccjit.int<-134> : !int
            gccjit.call @from_int_to_bitfield(%0, %1, %2, %3) : (!int, !int, !int, !int) -> ()
            gccjit.call @func_bitfield_to_int(%1, %2, %3, %0) : (!int, !int, !int, !int) -> ()

            %ret = gccjit.const #gccjit.zero : !int
            gccjit.return %ret : !int
    }
}
