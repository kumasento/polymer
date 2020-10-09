func @arith() {
  %A = alloc() : memref<32xf32>
  %B = alloc() : memref<32xf32>
  %C = alloc() : memref<32xf32>
  %D = alloc() : memref<32xf32>

  affine.for %i = 0 to 32 {
    %0 = affine.load %A[%i] : memref<32xf32>
    %1 = affine.load %B[%i] : memref<32xf32>
    %2 = addf %0, %1 : f32
    %3 = mulf %0, %2 : f32
    affine.store %2, %C[%i] : memref<32xf32>
    affine.store %3, %D[%i] : memref<32xf32>
  }
  
  return
}
