#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TOLERANCE 1e-5

// Define the memref struct
typedef struct {
  float *allocated;  // Pointer to allocated memory
  float *aligned;    // Pointer to aligned data (if applicable)
  size_t offset;     // Offset from aligned pointer to actual data
  size_t sizes[2];   // Array to hold sizes for the 2D dimensions
  size_t strides[2]; // Array to hold strides for the 2D dimensions
} memref_t;

// External declaration of the gemm function
extern void gemm(memref_t A, memref_t B, memref_t C, float alpha, float beta);

void initialize_matrix(memref_t *matrix, size_t rows, size_t cols) {
  matrix->allocated = (float *)malloc(rows * cols * sizeof(float));
  matrix->aligned = matrix->allocated; // Assume no special alignment needed
  matrix->offset = 0;
  matrix->sizes[0] = rows;
  matrix->sizes[1] = cols;
  matrix->strides[0] = cols; // Row-major layout
  matrix->strides[1] = 1;
}

void free_matrix(memref_t *matrix) { free(matrix->allocated); }

// Simple C implementation of gemm for verification
void gemm_verify(memref_t *A, memref_t *B, memref_t *C, float alpha,
                 float beta) {
  size_t rows = C->sizes[0];
  size_t cols = C->sizes[1];
  size_t K = A->sizes[1];

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; k++) {
        float a_val = A->allocated[i * A->strides[0] + k * A->strides[1]];
        float b_val = B->allocated[k * B->strides[0] + j * B->strides[1]];
        sum += a_val * b_val;
      }
      float c_val = C->allocated[i * C->strides[0] + j * C->strides[1]];
      C->allocated[i * C->strides[0] + j * C->strides[1]] =
          alpha * sum + beta * c_val;
    }
  }
}

// Function to check if two matrices are approximately equal
int verify_result(memref_t *C, memref_t *C_ref) {
  size_t rows = C->sizes[0];
  size_t cols = C->sizes[1];
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      float val = C->allocated[i * C->strides[0] + j * C->strides[1]];
      float ref_val =
          C_ref->allocated[i * C_ref->strides[0] + j * C_ref->strides[1]];
      if (fabs(val - ref_val) > TOLERANCE) {
        printf("Mismatch at C[%zu][%zu]: %f (expected %f)\n", i, j, val,
               ref_val);
        return 0;
      }
    }
  }
  return 1;
}

int main() {
  memref_t A, B, C, C_ref;
  float alpha = 1.0f, beta = 1.0f;
  size_t rows = 100, cols = 100;

  // Initialize matrices A, B, C, and C_ref with 100x100 dimensions
  initialize_matrix(&A, rows, cols);
  initialize_matrix(&B, rows, cols);
  initialize_matrix(&C, rows, cols);
  initialize_matrix(&C_ref, rows, cols);

  // Fill matrices A and B with some values and initialize C and C_ref
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      A.allocated[i * A.strides[0] + j * A.strides[1]] = (float)(i + j);
      B.allocated[i * B.strides[0] + j * B.strides[1]] = (float)(i - j);
      C.allocated[i * C.strides[0] + j * C.strides[1]] = 0.0f;
      C_ref.allocated[i * C_ref.strides[0] + j * C_ref.strides[1]] = 0.0f;
    }
  }

  // Call the external gemm function
  gemm(A, B, C, alpha, beta);

  // Call the verification gemm function
  gemm_verify(&A, &B, &C_ref, alpha, beta);

  // Verify the results
  if (verify_result(&C, &C_ref)) {
    printf("Verification passed! The matrices match.\n");
  } else {
    printf("Verification failed! The matrices do not match.\n");
  }

  // Free allocated memory for matrices
  free_matrix(&A);
  free_matrix(&B);
  free_matrix(&C);
  free_matrix(&C_ref);

  return 0;
}
