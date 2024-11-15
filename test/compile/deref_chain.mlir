// RUN: %gccjit-translate -o %t.gimple %s -mlir-to-gccjit-gimple
// RUN: %filecheck --input-file=%t.gimple %s --check-prefix=CHECK-GIMPLE
// RUN: %gccjit-translate -o %t.exe %s -mlir-to-gccjit-executable && chmod +x %t.exe 
// RUN: %t.exe | %filecheck %s --check-prefix=CHECK-OUTPUT

!int = !gccjit.int<int>
!bool = !gccjit.int<bool>
!ilv = !gccjit.lvalue<!int>
!vptr = !gccjit.ptr<!gccjit.void>
!size_t = !gccjit.int<size_t>
!cell = !gccjit.struct<"Cell" {
    #gccjit.field<"prev" !vptr>,
    #gccjit.field<"next" !vptr>,
    #gccjit.field<"data" !int>
}>
!str = !gccjit.ptr<!gccjit.qualified<!gccjit.int<char>, const>>

!list = !gccjit.struct<"List" {#gccjit.field<"dummy" !cell>}>

!cptr = !gccjit.ptr<!cell>
!lptr = !gccjit.ptr<!list>
!visitor = !gccjit.ptr<!gccjit.func<!gccjit.void (!cptr)>>

// CHECK-OUTPUT: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99
module @test attributes {
    gccjit.opt_level = #gccjit.opt_level<O3>, gccjit.debug_info = true, gccjit.cmdline_options = ["-fsanitize=address"], gccjit.driver_options = ["-fsanitize=address"]
} {
    gccjit.func internal @push_back(!lptr, !int) {
        ^entry(%arg0: !gccjit.lvalue<!lptr>, %arg1: !ilv):
            %0 = gccjit.sizeof !cell : !size_t
            %1 = gccjit.call builtin @__builtin_malloc(%0) : (!size_t) -> !vptr
            %2 = gccjit.as_rvalue %arg0 : !gccjit.lvalue<!lptr> to !lptr
            %3 = gccjit.as_rvalue %arg1 : !ilv to !int

            // fields of newly allocated cell
            // CHECK-GIMPLE: %[[NEW:[0-9]+]] = bitcast(%{{[0-9]+}}, struct Cell *);
            %4 = gccjit.bitcast %1 : !vptr to !cptr
            %5 = gccjit.deref_field %4[0] : !cptr -> !gccjit.lvalue<!vptr>
            %6 = gccjit.deref_field %4[1] : !cptr -> !gccjit.lvalue<!vptr>
            %7 = gccjit.deref_field %4[2] : !cptr -> !gccjit.lvalue<!int>

            // fields of dummy cell (prev)
            %8 = gccjit.deref_field %2[0] : !lptr -> !gccjit.lvalue<!cell>
            %9 = gccjit.access_field %8[0] : !gccjit.lvalue<!cell> -> !gccjit.lvalue<!vptr>
            // CHECK-GIMPLE: %[[TAIL:[0-9]+]] = %[[LIST:[0-9]+]]->dummy.prev;
            %10 = gccjit.as_rvalue %9 : !gccjit.lvalue<!vptr> to !vptr

            // fields of dummys's prev cell (next)
            // CHECK-GIMPLE: %[[CASTED_TAIL:[0-9]+]] = bitcast(%[[TAIL]], struct Cell *);
            %11 = gccjit.bitcast %10 : !vptr to !cptr
            // CHECK-GIMPLE: %[[DUMMY:[0-9]+]] = %[[CASTED_TAIL]]->next;
            %12 = gccjit.deref_field %11[1] : !cptr -> !gccjit.lvalue<!vptr>
            %13 = gccjit.as_rvalue %12 : !gccjit.lvalue<!vptr> to !vptr

            // update newly allocated cell
            // CHECK-GIMPLE: %[[NEW]]->data = %{{[0-9]+}};
            gccjit.assign %3 to %7 : !int, !ilv
            // CHECK-GIMPLE: %[[NEW]]->prev = %[[TAIL]];
            gccjit.assign %10 to %5 : !vptr, !gccjit.lvalue<!vptr>
            // CHECK-GIMPLE: %[[NEW]]->next = %[[DUMMY]];
            gccjit.assign %13 to %6 : !vptr, !gccjit.lvalue<!vptr>

            // update dummy cell
            // %[[LIST]]->dummy.prev = %[[NEW]];
            gccjit.assign %1 to %9 : !vptr, !gccjit.lvalue<!vptr>

            // update previous tail
            // %[[CASTED_TAIL]]->next = %[[NEW]];
            gccjit.assign %1 to %12 : !vptr, !gccjit.lvalue<!vptr>

            gccjit.return
    }

    gccjit.func internal @new_list() -> !lptr {
        %0 = gccjit.sizeof !list : !size_t
        %1 = gccjit.call builtin @__builtin_malloc(%0) : (!size_t) -> !vptr
        %2 = gccjit.bitcast %1 : !vptr to !lptr

        // initialize dummy
        %3 = gccjit.deref_field %2[0] : !lptr -> !gccjit.lvalue<!cell>
        %4 = gccjit.access_field %3[0] : !gccjit.lvalue<!cell> -> !gccjit.lvalue<!vptr>
        %5 = gccjit.access_field %3[1] : !gccjit.lvalue<!cell> -> !gccjit.lvalue<!vptr>
        %6 = gccjit.access_field %3[2] : !gccjit.lvalue<!cell> -> !gccjit.lvalue<!int>
        %8 = gccjit.const #gccjit.zero : !int
        gccjit.assign %1 to %4 : !vptr, !gccjit.lvalue<!vptr>
        gccjit.assign %1 to %5 : !vptr, !gccjit.lvalue<!vptr>
        gccjit.assign %8 to %6 : !int, !ilv

        gccjit.return %2 : !lptr
    }

    gccjit.func internal @_delete(!cptr) {
        ^entry(%arg0: !gccjit.lvalue<!cptr>):
            %0 = gccjit.as_rvalue %arg0 : !gccjit.lvalue<!cptr> to !cptr
            %1 = gccjit.bitcast %0 : !cptr to !vptr
            gccjit.call builtin @__builtin_free(%1) : (!vptr) -> ()
            gccjit.return
    }

    gccjit.func imported @printf(!str, ...) -> !int

    gccjit.func internal @_print(!cptr) {
        ^entry(%arg0: !gccjit.lvalue<!cptr>):
            %0 = gccjit.as_rvalue %arg0 : !gccjit.lvalue<!cptr> to !cptr
            %1 = gccjit.deref_field %0[2] : !cptr -> !gccjit.lvalue<!int>
            %2 = gccjit.as_rvalue %1 : !gccjit.lvalue<!int> to !int
            %3 = gccjit.literal "%d " : !str
            gccjit.call @printf(%3, %2) : (!str, !int) -> !int
            gccjit.return
    }

    gccjit.func internal @foreach(!lptr, !visitor) {
        ^entry(%arg0: !gccjit.lvalue<!lptr>, %arg1: !gccjit.lvalue<!visitor>):
            %0 = gccjit.as_rvalue %arg0 : !gccjit.lvalue<!lptr> to !lptr
            %1 = gccjit.as_rvalue %arg1 : !gccjit.lvalue<!visitor> to !visitor
            
            // get dummy's address
            // CHECK-GIMPLE: %[[ADDR:[0-9]+]] = &%[[LIST:[0-9]+]]->dummy;
            %2 = gccjit.deref_field %0[0] : !lptr -> !gccjit.lvalue<!cell>
            %3 = gccjit.addr (%2 : !gccjit.lvalue<!cell>) : !gccjit.ptr<!cell>

            // get dummy's next
            // CHECK-GIMPLE: %[[NEXT:[0-9]+]] = %[[LIST]]->dummy.next;
            %4 = gccjit.access_field %2[1] : !gccjit.lvalue<!cell> -> !gccjit.lvalue<!vptr>
            %5 = gccjit.as_rvalue %4 : !gccjit.lvalue<!vptr> to !vptr
            %6 = gccjit.bitcast %5 : !vptr to !cptr

            // initialize iterator
            %7 = gccjit.local : !gccjit.lvalue<!cptr>
            gccjit.assign %6 to %7 : !cptr, !gccjit.lvalue<!cptr>
            gccjit.jump ^loop.head

        ^loop.head:
            // check if iterator is dummy
            %8 = gccjit.compare eq (%7 : !gccjit.lvalue<!cptr>, %3 : !cptr) : !bool
            gccjit.conditional (%8 : !bool), ^loop.end, ^loop.body

        ^loop.body:
            // first load next
            %9 = gccjit.as_rvalue %7 : !gccjit.lvalue<!cptr> to !cptr
            %10 = gccjit.deref_field %9[1] : !cptr -> !gccjit.lvalue<!vptr>
            %11 = gccjit.as_rvalue %10 : !gccjit.lvalue<!vptr> to !vptr
            %12 = gccjit.bitcast %11 : !vptr to !cptr

            // then apply visitor
            gccjit.ptr_call %1(%9) : (!visitor, !cptr) -> !gccjit.void

            // then update iterator
            gccjit.assign %12 to %7 : !cptr, !gccjit.lvalue<!cptr>
            gccjit.jump ^loop.head

        ^loop.end:
            gccjit.return
    }

    gccjit.func internal @delete_list(!lptr) {
        ^entry(%arg0: !gccjit.lvalue<!lptr>):
            %0 = gccjit.as_rvalue %arg0 : !gccjit.lvalue<!lptr> to !lptr
            %1 = gccjit.fn_addr @_delete : !visitor
            gccjit.call @foreach(%0, %1) : (!lptr, !visitor) -> !gccjit.void
            %2 = gccjit.bitcast %0 : !lptr to !vptr
            gccjit.call builtin @__builtin_free(%2) : (!vptr) -> ()
            gccjit.return
    }

    gccjit.func internal @print_list(!lptr) {
        ^entry(%arg0: !gccjit.lvalue<!lptr>):
            %0 = gccjit.as_rvalue %arg0 : !gccjit.lvalue<!lptr> to !lptr
            // CHECK-GIMPLE: %{{[0-9]+}} = _print;
            %1 = gccjit.fn_addr @_print : !visitor
            gccjit.call @foreach(%0, %1) : (!lptr, !visitor) -> !gccjit.void
            %2 = gccjit.literal "\n" : !str
            gccjit.call @printf(%2) : (!str) -> !gccjit.void
            gccjit.return
    }

    gccjit.func exported @main() -> !int {
        %zero = gccjit.expr lazy { 
            %x = gccjit.const #gccjit.zero : !int
            gccjit.return %x : !int
        } : !int
        %0 = gccjit.call @new_list() : () -> !lptr
        %1 = gccjit.local : !ilv
        gccjit.assign %zero to %1 : !int, !ilv
        %2 = gccjit.const #gccjit.int<100> : !int
        gccjit.jump ^while.head

    ^while.head:
        %3 = gccjit.compare lt (%1 : !ilv, %2 : !int) : !bool
        gccjit.conditional (%3 : !bool), ^while.body, ^while.end

    ^while.body:
        %5 = gccjit.as_rvalue %1 : !ilv to !int
        gccjit.call @push_back(%0, %5) : (!lptr, !int) -> !gccjit.void
        %6 = gccjit.const #gccjit.one : !int
        // CHECK-GIMPLE: %{{[0-9]+}} += %{{[0-9]+}};
        gccjit.update plus %6 to %1 : !int, !ilv
        gccjit.jump ^while.head
    
    ^while.end:
        gccjit.call @print_list(%0) : (!lptr) -> !gccjit.void
        gccjit.call @delete_list(%0) : (!lptr) -> !gccjit.void
        gccjit.return %zero : !int
    }
} 
