func @load_store_2d(%A : memref<?x?xf32>, %B : memref<?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  %d0 = dim %A, %c0 : memref<?x?xf32>
  %d1 = dim %A, %c1 : memref<?x?xf32>

  affine.for %i = 0 to %d0 {
    affine.for %j = 0 to %d1 {
      %0 = affine.load %A[%i, %j] : memref<?x?xf32>
      affine.store %0, %B[%i, %j] : memref<?x?xf32>
    }
  }

  return 
}
