// RUN: polymer-opt %s -extract-scop-stmt | FileCheck %s
// RUN: polymer-opt %s -extract-scop-stmt -test-osl-scop-builder | FileCheck %s --check-prefix=BUILDER
// RUN: polymer-opt %s -extract-scop-stmt | polymer-translate -export-scop | FileCheck %s --check-prefix=OSL

// CHECK-LABEL: func @load_store_param
// BUILDER-LABEL: func @load_store_param
func @load_store_param(%A: memref<?xf32>) -> () {
  %c0 = constant 0 : index
  %N = dim %A, %c0 : memref<?xf32>

  affine.for %i = 0 to %N {
    %0 = affine.load %A[%N - %i - 1] : memref<?xf32>
    affine.store %0, %A[%i] : memref<?xf32>
  }

  return
}

// OSL-LABEL: <OpenScop>
