// File name: main.c

#include <stdio.h>

struct TwoDMemrefF32 {
  float *ptrToData;
  float *alignedPtrToData;
  long offset;
  long shape[2];
  long stride[2];
};

#define M 6
#define N 8

extern void _mlir_ciface_load_store_2d(struct TwoDMemrefF32 *,
                                       struct TwoDMemrefF32 *);

int main(int argc, char *argv[]) {
  int i, j;

  float A[M][N], B[M][N];

  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      A[i][j] = ((float)i + j) / (i + j + 1);
      B[i][j] = (float)0;
      printf("%8.6f ", A[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  struct TwoDMemrefF32 A_mem = {&A[0][0], &A[0][0], 0, {M, N}, {1, 1}};
  struct TwoDMemrefF32 B_mem = {&B[0][0], &B[0][0], 0, {M, N}, {1, 1}};

  _mlir_ciface_load_store_2d(&A_mem, &B_mem);

  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++)
      printf("%8.6f ", B[i][j]);
    printf("\n");
  }

  return 0;
}
