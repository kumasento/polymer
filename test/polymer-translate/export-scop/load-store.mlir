// RUN: polymer-opt %s -extract-scop-stmt | FileCheck %s
// RUN: polymer-opt %s -extract-scop-stmt -test-osl-scop-builder | FileCheck %s --check-prefix=BUILDER
// RUN: polymer-opt %s -extract-scop-stmt | polymer-translate -export-scop | FileCheck %s --check-prefix=OSL

// BUILDER: #[[SET:.*]] = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>

func @load_store() {
  %A = alloc() : memref<32xf32>
  // BUILDER-LABEL: %{{.*}} = alloc()
  // BUILDER: scop.param_names = ["A1"]

  affine.for %i = 0 to 32 {
    // BUILDER-LABEL: call @S0
    // BUILDER: scop.domain = #[[SET]]
    // BUILDER: scop.domain_symbols = ["i1"]
    %0 = affine.load %A[%i] : memref<32xf32>
    affine.store %0, %A[%i] : memref<32xf32>
  }
  // BUILDER: scop.iv_name = "i1"

  return
}


// CHECK-LABEL: func @S0
// CHECK: attributes {scop.stmt}
// CHECK: %[[VAL0:.*]] = affine.load
// CHECK: affine.store %[[VAL0]]

// OSL-LABEL: <OpenScop>

// OSL-LABEL: CONTEXT
// OSL: 0 2 0 0 0 0

// OSL: # Number of statements
// OSL-NEXT: 1

// OSL: # Number of relations describing the statement:
// OSL-NEXT: 4

// OSL-LABEL: DOMAIN
// OSL: 2 3 1 0 0 0
// OSL: # e/i| i1 |  1  
// OSL:    1    1    0    ## i1 >= 0
// OSL:    1   -1   31    ## -i1+31 >= 0

// OSL-LABEL: SCATTERING
// OSL: 3 6 3 1 0 0
// OSL: # e/i| c1   c2   c3 | i1 |  1  
// OSL:    0   -1    0    0    0    0    ## c1 == 0
// OSL:    0    0   -1    0    1    0    ## c2 == i1
// OSL:    0    0    0   -1    0    0    ## c3 == 0

// OSL-LABEL: READ
// OSL: 2 5 2 1 0 0
// OSL: # e/i| Arr  [1]| i1 |  1  
// OSL:    0    0   -1    1    0    ## [1] == i1
// OSL:    0   -1    0    0    1    ## Arr == A1

// OSL-LABEL: WRITE
// OSL: 2 5 2 1 0 0
// OSL: # e/i| Arr  [1]| i1 |  1  
// OSL:    0    0   -1    1    0    ## [1] == i1
// OSL:    0   -1    0    0    1    ## Arr == A1

// OSL-LABEL: <body>
// OSL: i1
// OSL: S0(i1,A1)

// OSL-LABEL: <scatnames>
// OSL: c1 c2 c3

// OSL-LABEL: <arrays>
// OSL: 1 A1
