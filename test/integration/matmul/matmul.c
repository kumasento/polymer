#include <math.h>
#include <stdio.h>

#include "polybench.h"

#define NI 4
#define NJ 4
#define NK 4

#define DATA_TYPE float

// MLIR interface
struct TwoDMemrefF32 {
  DATA_TYPE *ptrToData;
  DATA_TYPE *alignedPtrToData;
  float offset;
  long shape[2];
  long stride[2];
};

#define MEMREF TwoDMemrefF32
#define kernel_mlir_plain _mlir_ciface_matmul

extern void _mlir_ciface_matmul(struct TwoDMemrefF32 *, struct TwoDMemrefF32 *,
                                struct TwoDMemrefF32 *);

static void init_array(DATA_TYPE A[NI][NK], DATA_TYPE B[NK][NJ],
                       DATA_TYPE C[NI][NJ]) {
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NK; j++)
      A[i][j] = (DATA_TYPE)(i * j % NI) / NI;
  for (i = 0; i < NK; i++)
    for (j = 0; j < NJ; j++)
      B[i][j] = (DATA_TYPE)(i * (j + 1) % NJ) / NJ;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      C[i][j] = (DATA_TYPE)0;
}

static unsigned compare_array(DATA_TYPE C1[NI][NJ], DATA_TYPE C2[NI][NJ]) {
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      if (fabsf(C1[i][j] - C2[i][j]) >= 1e-6) {
        printf("i = %3d j = %3d C1 = %f C2 = %f\n", i, j, C1[i][j], C2[i][j]);
        return 0;
      }

  return 1;
}

static void kernel_plain(DATA_TYPE A[NI][NK], DATA_TYPE B[NK][NJ],
                         DATA_TYPE C[NI][NJ]) {
  int i, j, k;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i][j] = (DATA_TYPE)0;
      for (k = 0; k < NK; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

int main(int argc, char *argv[]) {
  int i, j;

  DATA_TYPE A[NI][NK];
  DATA_TYPE B[NK][NJ];
  DATA_TYPE C[NI][NJ];

  DATA_TYPE A_[NI * NK];
  DATA_TYPE B_[NJ * NK];
  DATA_TYPE C_[NI * NJ];

  struct MEMREF A_memref = {A_, A_, 0, {NI, NK}, {1, 1}};
  struct MEMREF B_memref = {B_, B_, 0, {NK, NJ}, {1, 1}};
  struct MEMREF C_memref = {C_, C_, 0, {NI, NJ}, {1, 1}};

  init_array(A, B, C);

  for (i = 0; i < NI; i++)
    for (j = 0; j < NK; j++)
      A_[i * NK + j] = A[i][j];
  for (i = 0; i < NK; i++)
    for (j = 0; j < NJ; j++)
      B_[i * NJ + j] = B[i][j];
  memset(C_, 0, sizeof(DATA_TYPE) * NI * NJ);

  printf("Running kernel_plain ...\n");

  /* Start timer. */
  polybench_start_instruments;

  kernel_plain(A, B, C);

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      printf("%8.6f ", C[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  kernel_mlir_plain(&A_memref, &B_memref, &C_memref);

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NK; j++) {
      printf("%8.6f ", A_[i * NK + j]);
    }
    printf("\n");
  }
  printf("\n");

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      printf("%8.6f ", C_[i * NJ + j]);
    }
    printf("\n");
  }

  printf("Finished running kernel MLIR plain ...\n");

  // if (compare_array(C1, C2))
  //   printf("TEST PASSED\n");
  // else
  //   printf("TEST FAILED\n");

  return 0;
}
