// RUN: polymer-opt %s -extract-scop-stmt -split-input-file | FileCheck %s

func @load_store(%A: memref<?xf32>, %B: memref<?xf32>) {
  %c0 = constant 0 : index
  %N = dim %A, %c0 : memref<?xf32> 

  affine.for %i = 0 to %N {
    %0 = affine.load %A[%i] : memref<?xf32>
    %1 = mulf %0, %0 : f32
    affine.store %1, %B[%i] : memref<?xf32>
  }

  return
}

// CHECK: func @load_store(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>) {
// CHECK-NEXT:   %[[C0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[DIM0:.*]] = dim %[[ARG0]], %[[C0]] : memref<?xf32>
// CHECK-NEXT:   affine.for %[[ARG2:.*]] = 0 to %[[DIM0]] {
// CHECK-NEXT:     call @S0(%[[ARG0]], %[[ARG2]], %[[ARG1]]) : (memref<?xf32>, index, memref<?xf32>) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK: func @S0(%[[ARG0]]: memref<?xf32>, %[[ARG1]]: index, %[[ARG2]]: memref<?xf32>) {
// CHECK-NEXT:   %[[VAL0:.*]] = affine.load %[[ARG0]][%[[ARG1]]] : memref<?xf32>
// CHECK-NEXT:   %[[VAL1:.*]] = mulf %[[VAL0]], %[[VAL0]] : f32
// CHECK-NEXT:   affine.store %[[VAL1]], %[[ARG2]][%[[ARG1]]] : memref<?xf32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// -----

func @load_multi_stores(%A: memref<?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %N = dim %B, %c0 : memref<?x?xf32> 
  %M = dim %B, %c1 : memref<?x?xf32> 

  affine.for %i = 0 to %N {
    affine.for %j = 0 to %M {
      %0 = affine.load %A[%i] : memref<?xf32>
      %1 = mulf %0, %0 : f32
      affine.store %1, %B[%i, %j] : memref<?x?xf32>
      %2 = addf %0, %1 : f32
      affine.store %2, %C[%i, %j] : memref<?x?xf32>
    }
  }

  return
}

// CHECK: func @load_multi_stores(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<?x?xf32>) {
// CHECK-NEXT:   %[[CST0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[CST1:.*]] = constant 1 : index
// CHECK-NEXT:   %[[DIM0:.*]] = dim %[[ARG1]], %[[CST0]] : memref<?x?xf32>
// CHECK-NEXT:   %[[DIM1:.*]] = dim %[[ARG1]], %[[CST1]] : memref<?x?xf32>
// CHECK-NEXT:   affine.for %[[I:.*]] = 0 to %[[DIM0]] {
// CHECK-NEXT:     affine.for %[[J:.*]] = 0 to %[[DIM1]] {
// CHECK-NEXT:       call @S0(%[[ARG0]], %[[I]], %[[ARG1]], %[[J]]) : (memref<?xf32>, index, memref<?x?xf32>, index) -> ()
// CHECK-NEXT:       call @S1(%[[ARG0]], %[[I]], %[[ARG2]], %[[J]]) : (memref<?xf32>, index, memref<?x?xf32>, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK: func @S0(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: memref<?x?xf32>, %[[ARG3:.*]]: index) {
// CHECK-NEXT:   %[[VAL0:.*]] = affine.load %[[ARG0]][%[[ARG1]]] : memref<?xf32>
// CHECK-NEXT:   %[[VAL1:.*]] = mulf %[[VAL0]], %[[VAL0]] : f32
// CHECK-NEXT:   affine.store %[[VAL1]], %[[ARG2]][%[[ARG1]], %[[ARG3]]] : memref<?x?xf32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK: func @S1(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: memref<?x?xf32>, %[[ARG3:.*]]: index) {
// CHECK-NEXT:   %[[VAL0:.*]] = affine.load %[[ARG0]][%[[ARG1]]] : memref<?xf32>
// CHECK-NEXT:   %[[VAL1:.*]] = mulf %[[VAL0]], %[[VAL0]] : f32
// CHECK-NEXT:   %[[VAL2:.*]] = addf %[[VAL0]], %[[VAL1]] : f32
// CHECK-NEXT:   affine.store %[[VAL2]], %[[ARG2]][%[[ARG1]], %[[ARG3]]] : memref<?x?xf32>
// CHECK-NEXT:   return
// CHECK-NEXT: }
