// RUN: polymer-opt %s -extract-scop-stmt | FileCheck %s
// RUN: polymer-opt %s -extract-scop-stmt -test-osl-scop-builder | FileCheck %s --check-prefix=BUILD
// RUN: polymer-opt %s -extract-scop-stmt | polymer-translate -export-scop | FileCheck %s --check-prefix=OSL

// Consider if operations in the domain.
// We will make this test valid when the diff D86421 is landed.

#set1 = affine_set<(d0, d1): (d0 - 16 >= 0, d1 - 16 >= 0, d1 - d0 >= 0)>
#set2 = affine_set<(d0): (- d0 + 31 == 0)>

// BUILD-DAG: #[[SET0:.*]] = affine_set<() : (0 == 0)>
// BUILD-DAG: #[[SET1:.*]] = affine_set<(d0, d1) : (d0 - 16 >= 0, d1 - 16 >= 0, d1 - d0 >= 0)>
// BUILD-DAG: #[[SET2:.*]] = affine_set<(d0, d1) : (-d1 + 31 >= 0, d0 - 16 >= 0, -d0 + d1 >= 0)>
// BUILD-DAG: #[[SET3:.*]] = affine_set<(d0) : (-d0 + 31 == 0)>
// BUILD-DAG: #[[SET4:.*]] = affine_set<(d0, d1) : (-d0 + 31 == 0, d1 >= 0, -d1 + 31 >= 0)>
// BUILD-DAG: #[[SET5:.*]] = affine_set<(d0, d1, d2, d3, d4) : (d1 - d3 == 0, d2 - d4 == 0, d0 - 2 == 0)>
// BUILD-DAG: #[[SET6:.*]] = affine_set<(d0, d1, d2, d3, d4) : (d1 - d3 == 0, d2 - d4 == 0, d0 - 1 == 0)>

// CHECK-LABEL: func @load_store_if
// BUILD-LABEL: func @load_store_if
func @load_store_if(%A : memref<32x32xf32>, %B : memref<32x32xf32>) -> () {
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to 32 {
      affine.if #set1(%i, %j) {
        // BUILD-LABEL: call @S0
        // BUILD: scop.domain = #[[SET2]]
        // BUILD: scop.domain_symbols = ["i1", "i2"]
        // BUILD: scop.scats = [0, 0, 0]
        %0 = affine.load %A[%i, %j] : memref<32x32xf32>
        affine.store %0, %B[%i, %j] : memref<32x32xf32>
      }
    }
    // BUILD: {scop.iv_name = "i2"}

    affine.if #set2(%i) {
      affine.for %j = 0 to 32 {
        // BUILD-LABEL: call @S1
        // BUILD: scop.domain = #[[SET4]]
        // BUILD: scop.domain_symbols = ["i1", "i3"]
        // BUILD: scop.scats = [0, 1, 0]
        %0 = affine.load %A[%i, %j] : memref<32x32xf32>
        %1 = addf %0, %0 : f32
        affine.store %1, %B[%i, %j] : memref<32x32xf32>
      }
      // BUILD: {scop.iv_name = "i3"}
    }
  }
  // BUILD: {scop.iv_name = "i1"}

  return
}

// BUILD-LABEL: func @S0
// BUILD-LABEL: affine.load
// BUILD: scop.access = #[[SET5]]
// BUILD: scop.access_symbols = ["A2", "", "", "i1", "i2"]
// BUILD-LABEL: affine.store
// BUILD: scop.access = #[[SET6]]
// BUILD: scop.access_symbols = ["A1", "", "", "i1", "i2"]

// BUILD-LABEL: func @S1
// BUILD-LABEL: affine.load
// BUILD: scop.access = #[[SET5]]
// BUILD: scop.access_symbols = ["A2", "", "", "i1", "i3"]
// BUILD-LABEL: affine.store
// BUILD: scop.access = #[[SET6]]
// BUILD: scop.access_symbols = ["A1", "", "", "i1", "i3"]


// OSL-LABEL: <OpenScop>

// OSL-LABEL: Statement 1

// OSL-LABEL: DOMAIN
// OSL: 3 4 2 0 0 0
// OSL: # e/i| i1   i2 |  1  
// OSL:    1    0   -1   31    ## -i2+31 >= 0
// OSL:    1    1    0  -16    ## i1-16 >= 0
// OSL:    1   -1    1    0    ## -i1+i2 >= 

// OSL-LABEL: SCATTERING
// OSL: 5 9 5 2 0 0
// OSL: # e/i| c1   c2   c3   c4   c5 | i1   i2 |  1  
// OSL:    0   -1    0    0    0    0    0    0    0    ## c1 == 0
// OSL:    0    0   -1    0    0    0    1    0    0    ## c2 == i1
// OSL:    0    0    0   -1    0    0    0    0    0    ## c3 == 0
// OSL:    0    0    0    0   -1    0    0    1    0    ## c4 == i2
// OSL:    0    0    0    0    0   -1    0    0    0    ## c5 == 0

// OSL-LABEL: Statement 2

// OSL-LABEL: DOMAIN
// OSL: 3 4 2 0 0 0
// OSL: # e/i| i1   i3 |  1  
// OSL:    0   -1    0   31    ## i1 == 31
// OSL:    1    0    1    0    ## i3 >= 0
// OSL:    1    0   -1   31    ## -i3+31 >= 0

// OSL-LABEL: SCATTERING
// OSL: 5 9 5 2 0 0
// OSL: # e/i| c1   c2   c3   c4   c5 | i1   i3 |  1  
// OSL:    0   -1    0    0    0    0    0    0    0    ## c1 == 0
// OSL:    0    0   -1    0    0    0    1    0    0    ## c2 == i1
// OSL:    0    0    0   -1    0    0    0    0    1    ## c3 == 1
// OSL:    0    0    0    0   -1    0    0    1    0    ## c4 == i3
// OSL:    0    0    0    0    0   -1    0    0    0    ## c5 == 0
