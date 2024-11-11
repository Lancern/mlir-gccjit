// RUN: %gccjit-opt %s \
// RUN:     -lower-affine \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-arith-to-gccjit \
// RUN:     -convert-memref-to-gccjit \
// RUN:     -convert-func-to-gccjit \
// RUN:     -reconcile-unrealized-casts -mlir-print-debuginfo -o %t.mlir 
// RUN: %filecheck --input-file=%t.mlir %s
// RUN: %gccjit-translate %t.mlir -mlir-to-gccjit-gimple | %filecheck %s --check-prefix=CHECK-GIMPLE
// RUN: %gccjit-translate %t.mlir -mlir-to-gccjit-object -o %t.o 
// RUN: nm %t.o | %filecheck %s --check-prefix=CHECK-NM


// CHECK-NM: T get_gv0
// CHECK-NM: T get_splat
// CHECK-NM: T get_uninit
// CHECK-NM: d gv0
// CHECK-NM: D gv1
// CHECK-NM: d splat
// CHECK-NM: T sum_uninit
// CHECK-NM: b uninit

module attributes { gccjit.opt_level = #gccjit.opt_level<O3>, gccjit.cmdline_options = ["-ffast-math"] } {
    // CHECK: gccjit.global internal  @splat init
    // CHECK-GIMPLE: static float[4][4] splat = (float[4][4]) {(float[4]) {(float)1.000000, (float)1.000000, (float)1.000000, (float)1.000000}, (float[4]) {(float)1.000000, (float)1.000000, (float)1.000000, (float)1.000000}, (float[4]) {(float)1.000000, (float)1.000000, (float)1.000000, (float)1.000000}, (float[4]) {(float)1.000000, (float)1.000000, (float)1.000000, (float)1.000000}};
	memref.global "private" @splat : memref<4x4xf32> = dense<1.0>
    // CHECK: gccjit.global internal  @gv0 init
    // CHECK-GIMPLE: static float[4] gv0 = (float[4]) {(float)0.000000, (float)1.000000, (float)2.000000, (float)3.000000};
	memref.global "private" @gv0 : memref<4xf32> = dense<[0.0, 1.0, 2.0, 3.0]>
    // CHECK: gccjit.global exported  @gv1 init
    // CHECK-GIMPLE: __uint32_t[2][3] gv1 = (__uint32_t[2][3]) {(__uint32_t[2]) {(__uint32_t)0, (__uint32_t)1}, (__uint32_t[2]) {(__uint32_t)2, (__uint32_t)3}, (__uint32_t[2]) {(__uint32_t)4, (__uint32_t)5}};
	memref.global @gv1 : memref<3x2xi32> = dense<[[0, 1],[2, 3],[4, 5]]>
    // CHECK: gccjit.global internal  @uninit
    // CHECK-GIMPLE: static float[512][512] uninit;
	memref.global "private" @uninit : memref<512x512xf32> = uninitialized


	func.func @get_splat() -> memref<4x4xf32> {
		%0 = memref.get_global @splat : memref<4x4xf32>
		return %0 : memref<4x4xf32>
	}

	func.func @get_gv0() -> memref<4xf32> {
		%0 = memref.get_global @gv0 : memref<4xf32>
		return %0 : memref<4xf32>
	}

	func.func @get_uninit() -> memref<512x512xf32> {
		%0 = memref.get_global @uninit : memref<512x512xf32>
		return %0 : memref<512x512xf32>
	}

	func.func @sum_uninit() -> f32 {
		%0 = memref.get_global @uninit : memref<512x512xf32>

		%zero = arith.constant 0.000000e+00 : f32

		%c0 = arith.constant 0 : index
		%c512 = arith.constant 512 : index
		%c1 = arith.constant 1 : index
		%sum = scf.for %arg0 = %c0 to %c512 step %c1 iter_args(%arg1 = %zero) -> f32 {
			%inner = scf.for %arg2 = %c0 to %c512 step %c1 iter_args(%arg3 = %arg1) -> f32 {
				%1 = memref.load %0[%arg0, %arg2] : memref<512x512xf32>
				%2 = arith.addf %arg3, %1 : f32
				scf.yield %2 : f32
			}
			scf.yield %inner : f32
		}
		return %sum : f32
	}
}
