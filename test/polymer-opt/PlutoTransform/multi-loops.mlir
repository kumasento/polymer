// RUN: polymer-opt %s -pluto-opt | FileCheck %s

func @multi_loops(%A : memref<?xf32>) {
  %c0 = constant 0 : index
  %cst = constant 1.0 : f32

  %d0 = dim %A, %c0 : memref<?xf32>

  affine.for %i = 0 to %d0 {
    affine.store %cst, %A[%i] : memref<?xf32>
  }

  affine.for %i = 0 to %d0 {
    affine.store %cst, %A[%i] : memref<?xf32>
  }

  return 
}

// CHECK: #map0 = affine_map<(d0) -> (d0)>
// CHECK: #map1 = affine_map<(d0) -> (d0 * 32)>
// CHECK: #map2 = affine_map<(d0)[s0] -> (s0 - 1, d0 * 32 + 31)>
// CHECK: #map3 = affine_map<() -> (0)>
// CHECK: #map4 = affine_map<()[s0] -> ((s0 - 1) floordiv 32)>
//
//
// CHECK: module {
// CHECK:   func @main(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: memref<?xf32>, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index) {
// CHECK:     affine.for %[[ARG5:.*]] = 0 to #map4()[%[[ARG0]]] {
// CHECK:       affine.for %[[ARG6:.*]] = #map1(%[[ARG5]]) to min #map2(%[[ARG5]])[%[[ARG0]]] {
// CHECK:         %[[CST:.*]] = constant 1.000000e+00 : f32
// CHECK:         affine.store %[[CST]], %[[ARG2]][%[[ARG6]]] : memref<?xf32>
// CHECK:       }
// CHECK:     }
// CHECK:     affine.for %[[ARG5:.*]] = 0 to #map4()[%[[ARG0]]] {
// CHECK:       affine.for %[[ARG6:.*]] = #map1(%[[ARG5]]) to min #map2(%[[ARG5]])[%[[ARG0]]] {
// CHECK:         %[[CST:.*]] = constant 1.000000e+00 : f32
// CHECK:         affine.store %[[CST]], %[[ARG2]][%[[ARG6]]] : memref<?xf32>
// CHECK:       }
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }
