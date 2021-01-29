// RUN: polymer-opt %s -reg2mem -extract-scop-stmt | polymer-translate -export-scop | FileCheck %s --check-prefix=OSL

func @transpose(%A : memref<?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %N = dim %A, %c0 : memref<?x?xf32>
  %M = dim %A, %c1 : memref<?x?xf32>

  affine.for %i = 0 to %N {
    affine.for %j = 0 to %M {
      %0 = affine.load %A[%i, %j] : memref<?x?xf32>
      affine.store %0, %A[%j, %i] : memref<?x?xf32>
    }
  }

  return
}

// OSL-LABEL: <OpenScop>

// OSL-LABEL: READ
// OSL: 3 9 3 2 0 2
// OSL: # e/i| Arr  [1]  [2]| i1   i2 | P1   P2 |  1  
// OSL:    0    0   -1    0    1    0    0    0    0    ## [1] == i1
// OSL:    0    0    0   -1    0    1    0    0    0    ## [2] == i2
// OSL:    0   -1    0    0    0    0    0    0    1    ## Arr == A1

// OSL-LABEL: WRITE
// OSL: 3 9 3 2 0 2
// OSL: # e/i| Arr  [1]  [2]| i1   i2 | P1   P2 |  1  
// OSL:    0    0   -1    0    0    1    0    0    0    ## [1] == i2
// OSL:    0    0    0   -1    1    0    0    0    0    ## [2] == i1
// OSL:    0   -1    0    0    0    0    0    0    1    ## Arr == A1
