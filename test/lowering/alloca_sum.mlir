// RUN: %gccjit-opt %s \
// RUN:     -lower-affine \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-arith-to-gccjit \
// RUN:     -convert-memref-to-gccjit \
// RUN:     -convert-func-to-gccjit \
// RUN:     -reconcile-unrealized-casts -mlir-print-debuginfo -o %t.mlir 

// RUN: %gccjit-translate %t.mlir -mlir-to-gccjit-gimple | %filecheck %s --check-prefix=CHECK-GIMPLE
// RUN: %gccjit-translate %t.mlir -mlir-to-gccjit-dylib -o %t.so
// RUN: cc -O3 %p/alloca_sum.c %t.so -Wl,-rpath,%T -o %t.exe
// RUN: seq 1 100 | %t.exe | %filecheck %s --check-prefix=CHECK-OUTPUT

// CHECK-OUTPUT: 5050
module attributes { gccjit.opt_level = #gccjit.opt_level<O3>, gccjit.debug_info = false } {
  // Import C standard library functions for I/O
  func.func private @read() -> i32 
  func.func private @print(%val: i32)

  func.func @main() -> i32 {
    // Allocate memory for the 100 integers array and initialize sum to 0
    // CHECK-GIMPLE: %{{[0-9]+}} = bitcast(alloca (%{{[0-9]+}}), __uint32_t *);
    %array = memref.alloca() : memref<100xi32>
    %sum_init = arith.constant 0 : i32

    // Loop to read 100 integers into the array
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index

    scf.for %i = %c0 to %c100 step %c1 {
      // Call scanf to read an integer
      // CHECK-GIMPLE: %{{[0-9]+}} = read ();
      %read = func.call @read() : () -> i32
      memref.store %read, %array[%i] : memref<100xi32>
    }

    // Loop to calculate the sum using iter_args
    %final_sum = scf.for %i = %c0 to %c100 step %c1 iter_args(%acc = %sum_init) -> (i32) {
      %elem = memref.load %array[%i] : memref<100xi32>
      %new_sum = arith.addi %acc, %elem : i32
      scf.yield %new_sum : i32
    }

    // Print the result using printf
    // CHECK-GIMPLE: (void)print (%{{[0-9]+}});
    func.call @print(%final_sum) : (i32) -> ()

    %c0_1 = arith.constant 0 : i32
    return %c0_1 : i32
  }
}
