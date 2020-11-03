func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %ni = dim %A, %c0 : memref<?x?xf32>
  %nj = dim %B, %c1 : memref<?x?xf32>
  %nk = dim %A, %c1 : memref<?x?xf32>

  affine.for %i = 0 to %ni {
    affine.for %j = 0 to %nj {
      affine.for %k = 0 to %nk {
        %0 = affine.load %A[%i, %k] : memref<?x?xf32>
        %1 = affine.load %B[%k, %j] : memref<?x?xf32>
        %2 = mulf %0, %1 : f32
        %3 = affine.load %C[%i, %j] : memref<?x?xf32>
        %4 = addf %2, %3 : f32
        affine.store %4, %C[%i, %j] : memref<?x?xf32>
        // affine.store %0, %C[%i, %j] : memref<?x?xf32>
      }
    }
  }

  return 
}
