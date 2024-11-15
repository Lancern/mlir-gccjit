// RUN: %gccjit-translate -o %t.gimple %s -mlir-to-gccjit-gimple
// RUN: %filecheck --input-file=%t.gimple %s
// RUN: %gccjit-translate -o %t.exe %s -mlir-to-gccjit-executable && chmod +x %t.exe && %t.exe
!struct0 = !gccjit.struct<"Lancern" {
    #gccjit.field<"professional" !gccjit.fp<float>>,
    #gccjit.field<"genius" !gccjit.int<unsigned long>>
}>

!struct1 = !gccjit.struct<"QuarticCat" {
    #gccjit.field<"excellent" !gccjit.fp<float>>,
    #gccjit.field<"magnificent" !gccjit.int<long>>
}>

!union = !gccjit.union<"Union" {
    #gccjit.field<"professional" !struct0>,
    #gccjit.field<"genius" !struct1>,
    #gccjit.field<"data" !gccjit.int<int>>
}>

#float = #gccjit.float<0.15625> : !gccjit.fp<float>
#int = #gccjit.int<-1> : !gccjit.int<long>
module attributes { gccjit.opt_level = #gccjit.opt_level<O3>, gccjit.allow_unreachable = true } {

gccjit.global internal @union_data  init {
    %e = gccjit.const #float
    %m = gccjit.const #int
    %qc = gccjit.new_struct [0, 1] [%e , %m] : (!gccjit.fp<float>, !gccjit.int<long>) -> !struct1
    %un = gccjit.new_union %qc at 1 : !struct1,  !union
    gccjit.return %un : !union
} : !gccjit.lvalue<!union>

gccjit.func exported @main() -> !gccjit.int<int> {
    ^entry:
        %0 = gccjit.get_global @union_data : !gccjit.lvalue<!union>
        // CHECK: %{{[0-9]+}} = union_data.professional.genius;
        %1 = gccjit.access_field %0[0] : !gccjit.lvalue<!union> -> !gccjit.lvalue<!struct0>
        %2 = gccjit.access_field %1[1] : !gccjit.lvalue<!struct0> -> !gccjit.lvalue<!gccjit.int<unsigned long>>
        %3 = gccjit.as_rvalue %2 : !gccjit.lvalue<!gccjit.int<unsigned long>> to !gccjit.int<unsigned long>
        %max = gccjit.const #gccjit.int<0xffffffffffffffff> : !gccjit.int<unsigned long>
        %eq0 = gccjit.compare eq (%3 : !gccjit.int<unsigned long>, %max : !gccjit.int<unsigned long>) : !gccjit.int<bool>
        gccjit.conditional (%eq0 : !gccjit.int<bool>), ^next, ^abort

    ^next:
        // CHECK: %[[RV:[0-9]+]] = union_data.data;
        %4 = gccjit.access_field %0[2] : !gccjit.lvalue<!union> -> !gccjit.lvalue<!gccjit.int<int>>
        %5 = gccjit.as_rvalue %4 : !gccjit.lvalue<!gccjit.int<int>> to !gccjit.int<int>
        %target = gccjit.const #gccjit.int<1042284544> : !gccjit.int<int>
        %eq1 = gccjit.compare eq (%5 : !gccjit.int<int>, %target : !gccjit.int<int>) : !gccjit.int<bool>
        gccjit.conditional (%eq1 : !gccjit.int<bool>), ^return, ^abort

    ^abort:
        gccjit.call builtin @__builtin_trap() : () -> !gccjit.void
        gccjit.jump ^abort

    ^return:
        %ret = gccjit.const #gccjit.zero : !gccjit.int<int>
        gccjit.return %ret : !gccjit.int<int>
}

}
