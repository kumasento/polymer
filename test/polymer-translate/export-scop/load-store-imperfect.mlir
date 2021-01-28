// RUN: polymer-opt %s -reg2mem -extract-scop-stmt | FileCheck %s
// RUN: polymer-opt %s -reg2mem -extract-scop-stmt -test-osl-scop-builder | FileCheck %s --check-prefix=BUILD
// RUN: polymer-opt %s -reg2mem -extract-scop-stmt | polymer-translate -export-scop | FileCheck %s --check-prefix=OSL

// CHECK-LABEL: func @load_store_imperfect
// BUILD-LABEL: func @load_store_imperfect
func @load_store_imperfect(%A : memref<?xf32>, %B : memref<?x?xf32>) -> () {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  %M = dim %B, %c0 : memref<?x?xf32>
  // BUILD: dim {scop.param_names = ["P1"]}
  %N = dim %B, %c1 : memref<?x?xf32>
  // BUILD: dim {scop.param_names = ["P2"]}

  affine.for %i = 0 to %M {
    // BUILD: %[[MEM:.*]] = alloca() {scop.param_names = ["A1"]}
    // CHECK: call @S0
    %0 = affine.load %A[%M - %i - 1] : memref<?xf32>
    affine.for %j = 0 to %N {
      // CHECK: call @S1
      affine.store %0, %B[%i, %N - %j - 1] : memref<?x?xf32>
    }
  }
  // BUILD: scop.iv_name = "i2"
  // BUILD: scop.iv_name = "i1"

  return 
}

// CHECK-LABEL: func @S0
// CHECK-LABEL: func @S1

// OSL-LABEL: <OpenScop>

// OSL-LABEL: # Parameters are provided
// OSL: 1
// OSL: <strings>
// OSL: P1 P2

// OSL-LABEL: # Number of statements
// OSL: 2

// OSL-LABEL: Statement 1
// OSL-LABEL: DOMAIN
// OSL: 2 5 1 0 0 2
// OSL: # e/i| i1 | P1   P2 |  1  
// OSL:    1    1    0    0    0    ## i1 >= 0
// OSL:    1   -1    1    0   -1    ## -i1+P1-1 >= 0

// OSL-LABEL: SCATTERING
// OSL: 3 8 3 1 0 2
// OSL: # e/i| c1   c2   c3 | i1 | P1   P2 |  1  
// OSL:    0   -1    0    0    0    0    0    0    ## c1 == 0
// OSL:    0    0   -1    0    1    0    0    0    ## c2 == i1
// OSL:    0    0    0   -1    0    0    0    0    ## c3 == 0

// OSL-LABEL: WRITE
// OSL: 2 7 2 1 0 2
// OSL: # e/i| Arr  [1]| i1 | P1   P2 |  1  
// OSL:    0    0   -1    0    0    0    0    ## [1] == 0
// OSL:    0   -1    0    0    0    0    1    ## Arr == A1

// OSL-LABEL: # Statement body expression
// OSL: S0(i1)

// OSL-LABEL: Statement 2
// OSL-LABEL: DOMAIN
// OSL: 4 6 2 0 0 2
// OSL: # e/i| i1   i2 | P1   P2 |  1  
// OSL:    1    1    0    0    0    0    ## i1 >= 0
// OSL:    1   -1    0    1    0   -1    ## -i1+P1-1 >= 0
// OSL:    1    0    1    0    0    0    ## i2 >= 0
// OSL:    1    0   -1    0    1   -1    ## -i2+P2-1 >= 0

// OSL-LABEL: SCATTERING
// OSL: 5 11 5 2 0 2
// OSL: # e/i| c1   c2   c3   c4   c5 | i1   i2 | P1   P2 |  1  
// OSL:    0   -1    0    0    0    0    0    0    0    0    0    ## c1 == 0
// OSL:    0    0   -1    0    0    0    1    0    0    0    0    ## c2 == i1
// OSL:    0    0    0   -1    0    0    0    0    0    0    1    ## c3 == 1
// OSL:    0    0    0    0   -1    0    0    1    0    0    0    ## c4 == i2
// OSL:    0    0    0    0    0   -1    0    0    0    0    0    ## c5 == 0

// OSL-LABEL: READ
// OSL: 2 8 2 2 0 2
// OSL: # e/i| Arr  [1]| i1   i2 | P1   P2 |  1  
// OSL:    0    0   -1    0    0    0    0    0    ## [1] == 0
// OSL:    0   -1    0    0    0    0    0    1    ## Arr == A1

// OSL-LABEL: # Statement body expression
// OSL: S1(i1, i2)
