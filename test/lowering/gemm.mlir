// RUN: %gccjit-opt %s \
// RUN:     -lower-affine \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-arith-to-gccjit \
// RUN:     -convert-memref-to-gccjit \
// RUN:     -convert-func-to-gccjit \
// RUN:     -reconcile-unrealized-casts -mlir-print-debuginfo -o %t.mlir 
// RUN: %filecheck --input-file=%t.mlir %s
// RUN: %gccjit-translate %t.mlir -mlir-to-gccjit-gimple | %filecheck %s --check-prefix=CHECK-GIMPLE
module {
  // CHECK-NOT: func.func
  // CHECK-NOT: func.return
  // CHECK-NOT: cf.cond_br
  // CHECK-NOT: cf.br
  func.func @gemm(%A: memref<100x100xf32>, %B: memref<100x100xf32>, %C: memref<100x100xf32>, %alpha: f32, %beta: f32) {
    affine.for %i = 0 to 100 {
      affine.for %j = 0 to 100 {
        // Load the value from C and scale it by beta
        %c_val = affine.load %C[%i, %j] : memref<100x100xf32>
        %c_scaled = arith.mulf %c_val, %beta : f32
        
        // Initialize the accumulator
        %acc0 = arith.constant 0.0 : f32
        %sum = affine.for %k = 0 to 100 iter_args(%acc = %acc0) -> f32 {
          // Load values from A and B
          // CHECK-GIMPLE: %{{[0-9]+}} = %{{[0-9\.a-z]+}}[(%{{[0-9]+}} * (size_t)100 + %{{[0-9]+}})]
          %a_val = affine.load %A[%i, %k] : memref<100x100xf32>
          // CHECK-GIMPLE: %{{[0-9]+}} = %{{[0-9\.a-z]+}}[(%{{[0-9]+}} * (size_t)100 + %{{[0-9]+}})]
          %b_val = affine.load %B[%k, %j] : memref<100x100xf32>
          
          // Multiply and accumulate
          // CHECK-GIMPLE: %[[V:[0-9]+]] = %{{[0-9]+}} * %{{[0-9]+}}
          %prod = arith.mulf %a_val, %b_val : f32
          // CHECK-GIMPLE: %{{[0-9]+}} = %{{[0-9]+}} + %[[V]]
          %new_acc = arith.addf %acc, %prod : f32
          
          // Yield the new accumulated value
          affine.yield %new_acc : f32
        }
        
        // Multiply the sum by alpha
        %result = arith.mulf %sum, %alpha : f32
        
        // Add the scaled C matrix value to the result
        %final_val = arith.addf %c_scaled, %result : f32
        
        // Store the final result back to matrix C
        // CHECK-GIMPLE: %{{[0-9\.a-z]+}}[(%{{[0-9]+}} * (size_t)100 + %{{[0-9]+}})] = %{{[0-9]+}}
        affine.store %final_val, %C[%i, %j] : memref<100x100xf32>
      }
    }
    return
  }
}
