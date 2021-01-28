// RUN: polymer-opt %s -reg2mem -extract-scop-stmt | FileCheck %s
// RUN: polymer-opt %s -reg2mem -extract-scop-stmt -test-osl-scop-builder | FileCheck %s --check-prefix=BUILD
// RUN: polymer-opt %s -reg2mem -extract-scop-stmt | polymer-translate -export-scop | FileCheck %s --check-prefix=OSL

// Consider local variables in the domain.

#map = affine_map<(d0) -> (d0 floordiv 2)>
// CHECK: #[[MAP:.*]] = affine_map<(d0) -> (d0 floordiv 2)>

// BUILD: #[[SET3:.*]] = affine_set<(d0, d1, d2)[s0] : (d1 - d2 == 0, d0 - 1 == 0)>
// BUILD: #[[SET4:.*]] = affine_set<(d0, d1, d2)[s0] : (d1 - d2 - s0 floordiv 2 == 0, d0 - 1 == 0, s0 mod 2 >= 0, -s0 + (s0 floordiv 2) * 2 + 1 >= 0)>
// BUILD: #[[SET5:.*]] = affine_set<(d0, d1, d2)[s0] : (d1 - d2 floordiv 2 - s0 floordiv 2 == 0, d0 - 1 == 0, d2 mod 2 >= 0, -d2 + (d2 floordiv 2) * 2 + 1 >= 0, s0 mod 2 >= 0, -s0 + (s0 floordiv 2) * 2 + 1 >= 0)>

// CHECK-LABEL: func @load_store_local_vars_floordiv
// BUILD-LABEL: func @load_store_local_vars_floordiv
func @load_store_local_vars_floordiv(%A : memref<?xf32>) -> () {
  %c0 = constant 0 : index
  // CHECK: %[[VAL0:.*]] = dim
  // BUILD: %[[VAL0:.*]] = dim
  %N = dim %A, %c0 : memref<?xf32>
  // BUILD: scop.param_names = ["P1"]

  // CHECK: %[[VAL1:.*]] = affine.apply
  // BUILD: %[[VAL1:.*]] = affine.apply
  %M = affine.apply #map(%N)
  
  // CHECK: affine.for %[[ARG1:.*]] = 0 to %[[VAL1]]
  // BUILD: affine.for %[[ARG1:.*]] = 0 to %[[VAL1]]
  affine.for %i = 0 to %M {
    %0 = affine.load %A[%i] : memref<?xf32>
    %j = affine.apply #map(%i)
    %1 = addf %0, %0 : f32
    affine.store %0, %A[%i + %M] : memref<?xf32>
    // CHECK: call @S0(%{{.*}}, %[[ARG1]], %[[VAL0]])
    // BUILD: call @S0(%{{.*}}, %[[ARG1]], %[[VAL0]])

    affine.store %1, %A[%j + %M] : memref<?xf32>
    // CHECK: call @S1(%{{.*}}, %[[VAL0]], %[[ARG1]])
    // BUILD: call @S1(%{{.*}}, %[[VAL0]], %[[ARG1]])
  }
  // BUILD: scop.iv_name = "i1"

  return
}

// CHECK-LABEL: func @S0
// CHECK: %[[VAL0:.*]] = affine.load
// CHECK: %[[VAL1:.*]] = affine.apply #[[MAP]]
// CHECK: affine.store %[[VAL0]], %{{.*}}[%{{.*}} + %[[VAL1]]]

// BUILD-LABEL: func @S0
// BUILD: scop.access = #[[SET3]]
// BUILD: scop.access = #[[SET4]]

// CHECK-LABEL: func @S1
// CHECK: %[[VAL0:.*]] = affine.load
// CHECK: %[[VAL1:.*]] = addf %[[VAL0]], %[[VAL0]]
// CHECK: %[[VAL2:.*]] = affine.apply #[[MAP]](%{{.*}})
// CHECK: %[[VAL3:.*]] = affine.apply #[[MAP]](%{{.*}})
// CHECK: affine.store %[[VAL1]], %{{.*}}[%[[VAL2]] + %[[VAL3]]]

// BUILD-LABEL: func @S1
// BUILD: scop.access = #[[SET3]]
// BUILD: scop.access = #[[SET5]]

// OSL-LABEL: <OpenScop>

// OSL-LABEL: # Context
// OSL: CONTEXT
// OSL: 0 3 0 0 0 1

// OSL-LABEL: # Parameters are provided
// OSL: 1
// OSL: <strings>
// OSL: P1
// OSL: </strings>

// OSL-LABEL: # Number of statements
// OSL: 2

// OSL-LABEL: Statement 1

// OSL-LABEL: DOMAIN
// OSL: 4 5 1 0 1 1
// OSL: # e/i| i1 | l1 | P1 |  1  
// OSL:    1    1    0    0    0    ## i1 >= 0
// OSL:    1    0   -2    1    0    ## -2*l1+P1 >= 0
// OSL:    1    0    2   -1    1    ## 2*l1-P1+1 >= 0
// OSL:    1   -1    1    0   -1    ## -i1+l1-1 >= 0

// OSL-LABEL: WRITE
// OSL: 4 8 2 1 2 1
// OSL: # e/i| Arr  [1]| i1 | l1   l2 | P1 |  1  
// OSL:    0    0   -1    1    1    0    0    0    ## [1] == i1+l1
// OSL:    0   -1    0    0    0    0    0    1    ## Arr == A1
// OSL:    1    0    0    0   -2    0    1    0    ## -2*l1+P1 >= 0
// OSL:    1    0    0    0    2    0   -1    1    ## 2*l1-P1+1 >= 0


// OSL-LABEL: Statement 2

// OSL-LABEL: DOMAIN
// OSL: 4 5 1 0 1 1
// OSL: # e/i| i1 | l1 | P1 |  1  
// OSL:    1    1    0    0    0    ## i1 >= 0
// OSL:    1    0   -2    1    0    ## -2*l1+P1 >= 0
// OSL:    1    0    2   -1    1    ## 2*l1-P1+1 >= 0
// OSL:    1   -1    1    0   -1    ## -i1+l1-1 >= 0


// OSL-LABEL: WRITE
// OSL: 6 9 2 1 3 1
// OSL: # e/i| Arr  [1]| i1 | l1   l2   l3 | P1 |  1  
// OSL:    0    0   -1    0    1    1    0    0    0    ## [1] == l1+l2
// OSL:    0   -1    0    0    0    0    0    0    1    ## Arr == A1
// OSL:    1    0    0    1   -2    0    0    0    0    ## i1-2*l1 >= 0
// OSL:    1    0    0   -1    2    0    0    0    1    ## -i1+2*l1+1 >= 0
// OSL:    1    0    0    0    0   -2    0    1    0    ## -2*l2+P1 >= 0
// OSL:    1    0    0    0    0    2    0   -1    1    ## 2*l2-P1+1 >= 0
