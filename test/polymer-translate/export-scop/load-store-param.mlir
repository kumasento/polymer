// RUN: polymer-opt %s -extract-scop-stmt | FileCheck %s
// RUN: polymer-opt %s -extract-scop-stmt -test-osl-scop-builder | FileCheck %s --check-prefix=BUILD
// RUN: polymer-opt %s -extract-scop-stmt | polymer-translate -export-scop | FileCheck %s --check-prefix=OSL

// BUILD: #[[SET:.*]] = affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>

// CHECK-LABEL: func @load_store_param
// BUILD-LABEL: func @load_store_param
func @load_store_param(%A: memref<?xf32>) -> () {
  %c0 = constant 0 : index
  %N = dim %A, %c0 : memref<?xf32>
  // BUILD: dim {scop.param_names = ["P1"]}

  affine.for %i = 0 to %N {
    %0 = affine.load %A[%N - %i - 1] : memref<?xf32>
    affine.store %0, %A[%i] : memref<?xf32>
    // BUILD-LABEL: call @S0
    // BUILD: scop.domain = #[[SET]]
    // BUILD: scop.domain_symbols = ["i1", "P1"]
  }
  // BUILD: {scop.iv_name = "i1"}

  return
}

// OSL-LABEL: <OpenScop>

// OSL-LABEL: # Context
// OSL-NEXT: CONTEXT
// OSL-NEXT: 0 3 0 0 0 1

// OSL-LABEL: # Parameters are provided
// OSL-NEXT: 1
// OSL-NEXT: <strings>
// OSL-NEXT: P1

// OSL-LABEL: DOMAIN
// OSL: 2 4 1 0 0 1
// OSL: # e/i| i1 | P1 |  1  
// OSL:    1    1    0    0    ## i1 >= 0
// OSL:    1   -1    1   -1    ## -i1+P1-1 >= 0

// OSL-LABEL: SCATTERING
// OSL: 3 7 3 1 0 1
// OSL: # e/i| c1   c2   c3 | i1 | P1 |  1  
// OSL:    0   -1    0    0    0    0    0    ## c1 == 0
// OSL:    0    0   -1    0    1    0    0    ## c2 == i1
// OSL:    0    0    0   -1    0    0    0    ## c3 == 0

// OSL-LABEL: READ
// OSL: 2 6 2 1 0 1
// OSL: # e/i| Arr  [1]| i1 | P1 |  1  
// OSL:    0    0   -1   -1    1   -1    ## [1] == -i1+P1-1
// OSL:    0   -1    0    0    0    1    ## Arr == A1

// OSL-LABEL: WRITE
// OSL: 2 6 2 1 0 1
// OSL: # e/i| Arr  [1]| i1 | P1 |  1  
// OSL:    0    0   -1    1    0    0    ## [1] == i1
// OSL:    0   -1    0    0    0    1    ## Arr == A1

