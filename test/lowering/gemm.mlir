module {
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
          %a_val = affine.load %A[%i, %k] : memref<100x100xf32>
          %b_val = affine.load %B[%k, %j] : memref<100x100xf32>
          
          // Multiply and accumulate
          %prod = arith.mulf %a_val, %b_val : f32
          %new_acc = arith.addf %acc, %prod : f32
          
          // Yield the new accumulated value
          affine.yield %new_acc : f32
        }
        
        // Multiply the sum by alpha
        %result = arith.mulf %sum, %alpha : f32
        
        // Add the scaled C matrix value to the result
        %final_val = arith.addf %c_scaled, %result : f32
        
        // Store the final result back to matrix C
        affine.store %final_val, %C[%i, %j] : memref<100x100xf32>
      }
    } { slap.extract }
    return
  }
}
