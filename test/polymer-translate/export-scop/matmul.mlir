// RUN: polymer-opt %s -reg2mem -extract-scop-stmt | FileCheck %s
// RUN: polymer-opt %s -reg2mem -extract-scop-stmt -test-osl-scop-builder | FileCheck %s --check-prefix=BUILD
// RUN: polymer-opt %s -reg2mem -extract-scop-stmt | polymer-translate -export-scop | FileCheck %s --check-prefix=OSL

// BUILD: #[[SET1:.*]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// BUILD: #[[SET2:.*]] = affine_set<(d0, d1, d2, d3, d4, d5) : (d1 - d3 == 0, d2 - d5 == 0, d0 - 3 == 0)>
// BUILD: #[[SET3:.*]] = affine_set<(d0, d1, d2, d3, d4, d5) : (d1 - d5 == 0, d2 - d4 == 0, d0 - 2 == 0)>
// BUILD: #[[SET4:.*]] = affine_set<(d0, d1, d2, d3, d4, d5) : (d1 - d3 == 0, d2 - d4 == 0, d0 - 1 == 0)>

// CHECK-LABEL: func @matmul
// BUILD-LABEL: func @matmul
func @matmul() {
  %A = alloc() : memref<64x64xf32>
  %B = alloc() : memref<64x64xf32>
  %C = alloc() : memref<64x64xf32>

  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      affine.for %k = 0 to 64 {
        %0 = affine.load %A[%i, %k] : memref<64x64xf32>
        %1 = affine.load %B[%k, %j] : memref<64x64xf32>
        %2 = mulf %0, %1 : f32
        %3 = affine.load %C[%i, %j] : memref<64x64xf32>
        %4 = addf %2, %3 : f32
        affine.store %4, %C[%i, %j] : memref<64x64xf32>

        // CHECK-LABEL: call @S0
        // BUILD-LABEL: call @S0
        // BUILD: scop.domain = #[[SET1]]
        // BUILD: scop.domain_symbols = ["i1", "i2", "i3"]
      }
      // BUILD: scop.iv_name = "i3"
    }
    // BUILD: scop.iv_name = "i2"
  }
  // BUILD: scop.iv_name = "i1"

  return
}

// CHECK-LABEL: func @S0
// CHECK: %[[VAL0:.*]] = affine.load
// CHECK-NEXT: %[[VAL1:.*]] = affine.load
// CHECK-NEXT: %[[VAL2:.*]] = mulf %[[VAL0]], %[[VAL1]]
// CHECK-NEXT: %[[VAL3:.*]] = affine.load
// CHECK-NEXT: %[[VAL4:.*]] = addf %[[VAL2]], %[[VAL3]]
// CHECK-NEXT: affine.store %[[VAL4]]

// BUILD-LABEL: func @S0
// BUILD: %[[VAL0:.*]] = affine.load 
// BUILD: scop.access = #[[SET2]], scop.access_symbols = ["A3", "", "", "i1", "i2", "i3"]
// BUILD: %[[VAL1:.*]] = affine.load 
// BUILD: scop.access = #[[SET3]], scop.access_symbols = ["A2", "", "", "i1", "i2", "i3"]
// BUILD: %[[VAL2:.*]] = mulf %[[VAL0]], %[[VAL1]]
// BUILD: %[[VAL3:.*]] = affine.load 
// BUILD: scop.access = #[[SET4]], scop.access_symbols = ["A1", "", "", "i1", "i2", "i3"]
// BUILD: %[[VAL4:.*]] = addf %[[VAL2]], %[[VAL3]]
// BUILD: affine.store %[[VAL4]]
// BUILD: scop.access = #[[SET4]], scop.access_symbols = ["A1", "", "", "i1", "i2", "i3"]

// OSL-LABEL: <OpenScop>

// OSL-LABEL: # Context
// OSL: CONTEXT
// OSL: 0 2 0 0 0 0

// OSL-LABEL: # Parameters are not provided
// OSL: 0

// OSL-LABEL: DOMAIN
// OSL: 6 5 3 0 0 0
// OSL: # e/i| i1   i2   i3 |  1  
// OSL:    1    1    0    0    0    ## i1 >= 0
// OSL:    1   -1    0    0   63    ## -i1+63 >= 0
// OSL:    1    0    1    0    0    ## i2 >= 0
// OSL:    1    0   -1    0   63    ## -i2+63 >= 0
// OSL:    1    0    0    1    0    ## i3 >= 0
// OSL:    1    0    0   -1   63    ## -i3+63 >= 0

// OSL-LABEL: SCATTERING
// OSL: 7 12 7 3 0 0
// OSL: # e/i| c1   c2   c3   c4   c5   c6   c7 | i1   i2   i3 |  1  
// OSL:    0   -1    0    0    0    0    0    0    0    0    0    0    ## c1 == 0
// OSL:    0    0   -1    0    0    0    0    0    1    0    0    0    ## c2 == i1
// OSL:    0    0    0   -1    0    0    0    0    0    0    0    0    ## c3 == 0
// OSL:    0    0    0    0   -1    0    0    0    0    1    0    0    ## c4 == i2
// OSL:    0    0    0    0    0   -1    0    0    0    0    0    0    ## c5 == 0
// OSL:    0    0    0    0    0    0   -1    0    0    0    1    0    ## c6 == i3
// OSL:    0    0    0    0    0    0    0   -1    0    0    0    0    ## c7 == 0

// OSL-LABEL: READ
// OSL: 3 8 3 3 0 0
// OSL: # e/i| Arr  [1]  [2]| i1   i2   i3 |  1  
// OSL:    0    0   -1    0    1    0    0    0    ## [1] == i1
// OSL:    0    0    0   -1    0    0    1    0    ## [2] == i3
// OSL:    0   -1    0    0    0    0    0    3    ## Arr == A3

// OSL-LABEL: READ
// OSL: 3 8 3 3 0 0
// OSL: # e/i| Arr  [1]  [2]| i1   i2   i3 |  1  
// OSL:    0    0   -1    0    0    0    1    0    ## [1] == i3
// OSL:    0    0    0   -1    0    1    0    0    ## [2] == i2
// OSL:    0   -1    0    0    0    0    0    2    ## Arr == A2

// OSL-LABEL: READ
// OSL: 3 8 3 3 0 0
// OSL: # e/i| Arr  [1]  [2]| i1   i2   i3 |  1  
// OSL:    0    0   -1    0    1    0    0    0    ## [1] == i1
// OSL:    0    0    0   -1    0    1    0    0    ## [2] == i2
// OSL:    0   -1    0    0    0    0    0    1    ## Arr == A1

// OSL-LABEL: WRITE
// OSL: 3 8 3 3 0 0
// OSL: # e/i| Arr  [1]  [2]| i1   i2   i3 |  1  
// OSL:    0    0   -1    0    1    0    0    0    ## [1] == i1
// OSL:    0    0    0   -1    0    1    0    0    ## [2] == i2
// OSL:    0   -1    0    0    0    0    0    1    ## Arr == A1

// OSL-LABEL: <body>
// OSL: # Number of original iterators
// OSL: 3
// OSL: # List of original iterators
// OSL: i1 i2 i3
// OSL: # Statement body expression
// OSL: S0(i1,i2,A1,i3,A2,A3)

// OSL-DAG: 1 A1
// OSL-DAG: 2 A2
// OSL-DAG: 3 A3
