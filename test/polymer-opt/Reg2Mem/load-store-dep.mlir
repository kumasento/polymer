// RUN: polymer-opt %s -reg2mem | polymer-opt | FileCheck %s

func @load_store_dep(%A: memref<?xf32>, %B: memref<?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  %someValue = constant 1.23 : f32

  %NI = dim %A, %c0 : memref<?xf32>
  %NJ = dim %B, %c1 : memref<?x?xf32>

  affine.for %i = 0 to %NI {
    %0 = affine.load %A[%i] : memref<?xf32>
    %1 = mulf %0, %0 : f32
    affine.store %someValue, %A[%i] : memref<?xf32>

    affine.for %j = 0 to %NJ {
      %2 = mulf %1, %0 : f32
      %3 = addf %1, %2 : f32
      %4 = subf %3, %someValue : f32
      affine.store %4, %B[%i, %j] : memref<?x?xf32>
    }

    affine.store %1, %A[%i] : memref<?xf32>
  }

  return 
}

// CHECK: func @load_store_dep(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?x?xf32>) {
// CHECK:   %[[C0:.*]] = constant 0 : index
// CHECK:   %[[C1:.*]] = constant 1 : index
// CHECK:   %[[CST:.*]] = constant 1.230000e+00 : f32
// CHECK:   %[[MEM0:.*]] = alloca() : memref<1xf32>
// CHECK:   affine.store %[[CST]], %[[MEM0]][0] : memref<1xf32>
// CHECK:   %[[DIM0:.*]] = dim %[[ARG0]], %[[C0]] : memref<?xf32>
// CHECK:   %[[DIM1:.*]] = dim %[[ARG1]], %[[C1]] : memref<?x?xf32>
// CHECK:   affine.for %[[I:.*]] = 0 to %[[DIM0]] {
// CHECK:     %[[VAL0:.*]] = affine.load %[[MEM0]][0] : memref<1xf32>
// CHECK:     %[[VAL1:.*]] = affine.load %[[ARG0]][%[[I]]] : memref<?xf32>
// CHECK:     %[[MEM1:.*]] = alloca() : memref<1xf32>
// CHECK:     affine.store %[[VAL1]], %[[MEM1]][0] : memref<1xf32>
// CHECK:     %[[VAL2:.*]] = mulf %[[VAL1]], %[[VAL1]] : f32
// CHECK:     %[[MEM2:.*]] = alloca() : memref<1xf32>
// CHECK:     affine.store %[[VAL2]], %[[MEM2]][0] : memref<1xf32>
// CHECK:     affine.store %[[VAL0]], %[[ARG0]][%[[I]]] : memref<?xf32>
// CHECK:     affine.for %[[J:.*]] = 0 to %[[DIM1]] {
// CHECK-DAG:   %[[VAL3:.*]] = affine.load %[[MEM0]][0] : memref<1xf32>
// CHECK-DAG:   %[[VAL4:.*]] = affine.load %[[MEM1]][0] : memref<1xf32>
// CHECK-DAG:   %[[VAL5:.*]] = affine.load %[[MEM2]][0] : memref<1xf32>
// CHECK:       %[[VAL6:.*]] = mulf %[[VAL5]], %[[VAL4]] : f32
// CHECK:       %[[VAL7:.*]] = addf %[[VAL5]], %[[VAL6]] : f32
// CHECK:       %[[VAL8:.*]] = subf %[[VAL7]], %[[VAL3]] : f32
// CHECK:       affine.store %[[VAL8]], %[[ARG1]][%[[I]], %[[J]]] : memref<?x?xf32>
// CHECK:     }
// CHECK:     affine.store %[[VAL2]], %[[ARG0]][%[[I]]] : memref<?xf32>
// CHECK:   }
// CHECK:   return
// CHECK: }
