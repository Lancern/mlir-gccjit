// RUN: %gccjit-translate -o %t.gimple %s -mlir-to-gccjit-gimple
// RUN: %filecheck --input-file=%t.gimple %s --check-prefix=CHECK-GIMPLE
// RUN: %gccjit-translate -o %t.exe %s -mlir-to-gccjit-executable && chmod +x %t.exe 
// RUN: %t.exe | %filecheck %s --check-prefix=CHECK-OUTPUT

!char = !gccjit.int<char>
!charptr = !gccjit.ptr<!char>
!char_vector = !gccjit.vector<!char, 16>
!size_t = !gccjit.int<size_t>
!bool = !gccjit.int<bool>
!str = !gccjit.ptr<!gccjit.qualified<!char, const>>

// CHECK-OUTPUT: abcABCEFLJASDVKMFsdadasjlfdjfldSDASFKSDAFNKDASFJasdasdadsdKLFDSFNSDLKFNASDKFKNagjrwtoDSADASDASD
module @vector attributes {
  gccjit.opt_level = #gccjit.opt_level<O3>,
  gccjit.prog_name = "test/compile/vector.mlir"
} {

    gccjit.func always_inline @alpha_flipcase_small(!charptr, !size_t) {
        ^entry(%0: !gccjit.lvalue<!charptr>, %1: !gccjit.lvalue<!size_t>):
            %c32 = gccjit.expr lazy { 
                %a = gccjit.const #gccjit.int<32> : !char
                gccjit.return %a : !char
            } : !char
            %zero = gccjit.expr lazy {
                %b = gccjit.const #gccjit.zero : !size_t
                gccjit.return %b : !size_t
            } : !size_t
            %one = gccjit.expr lazy { 
                %c = gccjit.const #gccjit.one : !size_t
                gccjit.return %c : !size_t
            } : !size_t
            %iter = gccjit.local : !gccjit.lvalue<!size_t>
            gccjit.assign %zero to %iter : !size_t, !gccjit.lvalue<!size_t>
            gccjit.jump ^loop.header
        
        ^loop.header:
            %cond = gccjit.expr lazy { 
                %flag = gccjit.compare eq (
                    %iter : !gccjit.lvalue<!size_t>, %1 : !gccjit.lvalue<!size_t>) : !bool
                gccjit.return %flag : !bool
            } : !bool
            gccjit.conditional (%cond : !bool), ^loop.exit, ^loop.body

        ^loop.body:
            %lv = gccjit.deref (%0 : !gccjit.lvalue<!charptr>, %iter: !gccjit.lvalue<!size_t>)
                : !gccjit.lvalue<!char>
            gccjit.update bitwise_xor %c32 to %lv : !char, !gccjit.lvalue<!char>
            gccjit.update plus %one to %iter : !size_t, !gccjit.lvalue<!size_t>
            gccjit.jump ^loop.header

        ^loop.exit:
            gccjit.return
    }

    gccjit.func exported @alpha_flipcase(!charptr, !size_t) {
        // CHECK-GIMPLE: char  __attribute__((vector_size(sizeof (char) * 16))) %[[VEC:[0-9]+]];
        ^entry(%0: !gccjit.lvalue<!charptr>, %1: !gccjit.lvalue<!size_t>):
            %c32 = gccjit.expr lazy { 
                %a = gccjit.const #gccjit.int<32> : !char
                gccjit.return %a : !char
            } : !char
            // CHECK-GIMPLE: %[[VEC]] = {(char)32, (char)32, (char)32, (char)32, (char)32, (char)32, (char)32, (char)32, (char)32, (char)32, (char)32, (char)32, (char)32, (char)32, (char)32, (char)32};
            %vec = gccjit.new_vector !char_vector [
                %c32, %c32, %c32, %c32, %c32, %c32, %c32, %c32,
                %c32, %c32, %c32, %c32, %c32, %c32, %c32, %c32
            ]
            %zero = gccjit.expr lazy {
                %b = gccjit.const #gccjit.zero : !size_t
                gccjit.return %b : !size_t
            } : !size_t
            %c16 = gccjit.expr lazy {
                %c = gccjit.const #gccjit.int<16> : !size_t
                gccjit.return %c : !size_t
            } : !size_t

            %iter = gccjit.local : !gccjit.lvalue<!size_t>
            gccjit.assign %zero to %iter : !size_t, !gccjit.lvalue<!size_t>
            gccjit.jump ^loop.header
        
        ^loop.header:
            %cond = gccjit.expr lazy {
                %diff = gccjit.binary minus (%1 : !gccjit.lvalue<!size_t>, %iter : !gccjit.lvalue<!size_t>) : !size_t
                %flag = gccjit.compare lt (%diff : !size_t, %c16 : !size_t) : !bool
                gccjit.return %flag : !bool
            } : !bool
            gccjit.conditional (%cond : !bool), ^loop.exit, ^loop.body

        ^loop.body:
            // CHECK-GIMPLE: %[[ADDR:[0-9]+]] = &%arg0[%{{[0-9]+}}];
            // CHECK-GIMPLE: %{{[0-9]+}} = (char  __attribute__((vector_size(sizeof (char) * 16))) *)%[[ADDR]];
            %cursor = gccjit.deref (%0 : !gccjit.lvalue<!charptr>, %iter: !gccjit.lvalue<!size_t>)
                : !gccjit.lvalue<!charptr>
            %cursor_addr = gccjit.addr (%cursor : !gccjit.lvalue<!charptr>) : !gccjit.ptr<!char>
            %cast = gccjit.cast %cursor_addr : !gccjit.ptr<!char> to !gccjit.ptr<!char_vector>
            %deref = gccjit.deref (%cast : !gccjit.ptr<!char_vector>)
                : !gccjit.lvalue<!char_vector>
            gccjit.update bitwise_xor %vec to %deref : !char_vector, !gccjit.lvalue<!char_vector>
            gccjit.update plus %c16 to %iter : !size_t, !gccjit.lvalue<!size_t>
            gccjit.jump ^loop.header

        ^loop.exit:
            %new_start = gccjit.deref (%0 : !gccjit.lvalue<!charptr>, %iter: !gccjit.lvalue<!size_t>)
                : !gccjit.lvalue<!charptr>
            %new_addr = gccjit.addr (%new_start : !gccjit.lvalue<!charptr>) : !gccjit.ptr<!char>
            %new_size = gccjit.binary minus (%1 : !gccjit.lvalue<!size_t>, %iter : !gccjit.lvalue<!size_t>)
                : !size_t
            gccjit.call @alpha_flipcase_small(%new_addr, %new_size) : (!gccjit.ptr<!char>, !size_t) -> ()
            gccjit.return
    }

    gccjit.func imported @strlen(!str) -> !size_t
    gccjit.func imported @puts(!str) -> !gccjit.int<int>

    gccjit.func exported @main () -> !gccjit.int<int> {
        %data = gccjit.literal "ABCabcefljasdvkmfSDADASJLFDJFLDsdasfksdafnkdasfjASDASDADSDklfdsfnsdlkfnasdkfknAGJRWTOdsadasdasd" : !str
        %one = gccjit.const #gccjit.one : !size_t
        %len = gccjit.call @strlen(%data) : (!str) -> !size_t
        %len_plus_one = gccjit.binary plus (%len : !size_t, %one : !size_t) : !size_t
        %alloca = gccjit.call builtin @alloca (%len_plus_one) : (!size_t) -> !gccjit.ptr<!gccjit.void>
        %voidptr = gccjit.bitcast %data : !str to !gccjit.ptr<!gccjit.qualified<!gccjit.void, const>>
        gccjit.call builtin @memcpy (%alloca, %voidptr, %len_plus_one) : (!gccjit.ptr<!gccjit.void>, !gccjit.ptr<!gccjit.qualified<!gccjit.void, const>>, !size_t) -> !gccjit.ptr<!gccjit.void>
        %charptr = gccjit.cast %alloca : !gccjit.ptr<!gccjit.void> to !gccjit.ptr<!char>
        gccjit.call @alpha_flipcase(%charptr, %len) : (!gccjit.ptr<!char>, !size_t) -> ()
        gccjit.call @puts(%charptr) : (!gccjit.ptr<!char>) -> !gccjit.int<int>
        %zero = gccjit.const #gccjit.zero : !gccjit.int<int>
        gccjit.return %zero : !gccjit.int<int>
    }
}
