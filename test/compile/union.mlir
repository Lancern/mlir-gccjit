// RUN: %gccjit-translate -o %t.gimple %s -mlir-to-gccjit-gimple
// RUN: %filecheck --input-file=%t.gimple %s
// RUN: %gccjit-translate -o %t.exe %s -mlir-to-gccjit-executable && chmod +x %t.exe && %t.exe
!struct1 = !gccjit.struct<"Lancern" {
    #gccjit.field<"professional" !gccjit.fp<float>>,
    #gccjit.field<"genius" !gccjit.int<unsigned long>>
}>

!struct2 = !gccjit.struct<"QuarticCat" {
    #gccjit.field<"excellent" !gccjit.fp<float>>,
    #gccjit.field<"magnificent" !gccjit.int<long>>
}>

!struct3 = !gccjit.struct<"Float" {
    #gccjit.field<"mant" !gccjit.int<uint32_t> : 23>,
    #gccjit.field<"exp" !gccjit.int<uint32_t> : 8>,
    #gccjit.field<"sign" !gccjit.int<uint32_t> : 1>
}>

!union = !gccjit.union<"Union" {
    #gccjit.field<"professional" !struct1>,
    #gccjit.field<"genius" !struct2>,
    #gccjit.field<"_float" !struct3>
}>

#float = #gccjit.float<0.15625> : !gccjit.fp<float>
#int = #gccjit.int<-1> : !gccjit.int<long>
module attributes { gccjit.opt_level = #gccjit.opt_level<O3>, gccjit.allow_unreachable = true } {

gccjit.global internal @union_data  init {
    %e = gccjit.const #float
    %m = gccjit.const #int
    %qc = gccjit.new_struct [0, 1] [%e , %m] : (!gccjit.fp<float>, !gccjit.int<long>) -> !struct2
    %un = gccjit.new_union %qc at 1 : !struct2,  !union
    gccjit.return %un : !union
} : !gccjit.lvalue<!union>

gccjit.func exported @main() -> !gccjit.int<int> {
    ^entry:
        %0 = gccjit.get_global @union_data : !gccjit.lvalue<!union>
        %rv = gccjit.as_rvalue %0 : !gccjit.lvalue<!union> to !union
        // CHECK: %[[CAST:[0-9]+]] = bitcast(%{{[0-9]+}}, struct Lancern);
        %1 = gccjit.bitcast %rv : !union to !struct1
        // CHECK: %{{[0-9]+}} = %[[CAST]].genius;
        %2 = gccjit.access_field %1[1] : !struct1 -> !gccjit.int<unsigned long>
        %max = gccjit.const #gccjit.int<-1> : !gccjit.int<unsigned long>
        %eq = gccjit.compare eq (%2 : !gccjit.int<unsigned long>, %max : !gccjit.int<unsigned long>) : !gccjit.int<bool>
        gccjit.conditional (%eq : !gccjit.int<bool>), ^next, ^abort

    ^next:
        // CHECK: %[[RV:[0-9]+]] = bitcast(%{{[0-9]+}}, struct Float);
        %rv2 = gccjit.access_field %rv[1] : !union -> !struct2
        %3 = gccjit.bitcast %rv2 : !struct2 to !struct3
        // CHECK: %{{[0-9]+}} = %[[RV]].sign:1;
        %4 = gccjit.access_field %3[2] : !struct3 -> !gccjit.int<uint32_t> // sign
        // CHECK: %{{[0-9]+}} = %[[RV]].exp:8;
        %5 = gccjit.access_field %3[1] : !struct3 -> !gccjit.int<uint32_t> // exp
        // CHECK: %{{[0-9]+}} = %[[RV]].mant:23;
        %6 = gccjit.access_field %3[0] : !struct3 -> !gccjit.int<uint32_t> // mant
        %c0 = gccjit.const #gccjit.int<0> : !gccjit.int<uint32_t>
        %exp = gccjit.const #gccjit.int<124> : !gccjit.int<uint32_t>
        %mant = gccjit.const #gccjit.int<2097152> : !gccjit.int<uint32_t>
        %eq0 = gccjit.compare eq (%4 : !gccjit.int<uint32_t>, %c0 : !gccjit.int<uint32_t>) : !gccjit.int<bool>
        %eq1 = gccjit.compare eq (%5 : !gccjit.int<uint32_t>, %exp : !gccjit.int<uint32_t>) : !gccjit.int<bool>
        %eq2 = gccjit.compare eq (%6 : !gccjit.int<uint32_t>, %mant : !gccjit.int<uint32_t>) : !gccjit.int<bool>
        %and0 = gccjit.binary logical_and (%eq0 : !gccjit.int<bool>, %eq1 : !gccjit.int<bool>) : !gccjit.int<bool>
        %and1 = gccjit.binary logical_and (%and0 : !gccjit.int<bool>, %eq2 : !gccjit.int<bool>) : !gccjit.int<bool>
        gccjit.conditional (%and1 : !gccjit.int<bool>), ^return, ^abort

    ^abort:
        gccjit.call builtin @__builtin_trap() : () -> !gccjit.void
        gccjit.jump ^abort

    ^return:
        %ret = gccjit.const #gccjit.zero : !gccjit.int<int>
        gccjit.return %ret : !gccjit.int<int>
}

}
