func @load_store() -> () {
  %A = alloc() : memref<32xf32>
  affine.for %i = 0 to 32 {
    %0 = affine.load %A[%i] : memref<32xf32>
    affine.store %0, %A[%i] : memref<32xf32>
  }
  return
}

// -----

func @load_store_2d() -> () {
  %A = alloc() : memref<32x32xf32>
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to 32 {
      %0 = affine.load %A[%i, %j] : memref<32x32xf32>
      affine.store %0, %A[%i, %j] : memref<32x32xf32>
    }
  }
  return
}

// -----

func @load_store_2d_two_arrays() -> () {
  %A = alloc() : memref<32x32xf32>
  %B = alloc() : memref<32x32xf32>
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to 32 {
      %0 = affine.load %A[%i, %j] : memref<32x32xf32>
      affine.store %0, %B[%i, %j] : memref<32x32xf32>
    }
  }
  return
}

// -----

func @load_store_with_parameter(%A : memref<?xf32>) -> () {
  %c0 = constant 0 : index
  %N = dim %A, %c0 : memref<?xf32>
  affine.for %i = 0 to %N {
    %0 = affine.load %A[%i] : memref<?xf32>
    affine.store %0, %A[%i] : memref<?xf32>
  }
  return
}