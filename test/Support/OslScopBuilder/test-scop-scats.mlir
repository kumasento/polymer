// RUN: polymer-opt %s -test-osl-scop-builder -verify-diagnostics -split-input-file | FileCheck %s

// Test the ScatTree building in OslScopBuilder.

// CHECK-LABEL: func @no_affine_for
func @no_affine_for(%A: memref<1xf32>) {
  // expected-remark@above {{Has OslScop: true}}
  // CHECK: scop.scats = [0]
  call @S0(): () -> ()
  // CHECK: scop.scats = [1]
  call @S1(): () -> ()
  return
}

func private @S0() attributes {scop.stmt}
func private @S1() attributes {scop.stmt}

// -----

// CHECK-LABEL: func @perfectly_nested
func @perfectly_nested(%A: memref<?x?x?xf32>) {
  // expected-remark@above {{Has OslScop: true}}
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index

  %d0 = dim %A, %c0 : memref<?x?x?xf32>
  %d1 = dim %A, %c1 : memref<?x?x?xf32>
  %d2 = dim %A, %c2 : memref<?x?x?xf32>

  affine.for %i = 0 to %d0 {
    affine.for %j = 0 to %d1 {
      affine.for %k = 0 to %d2 {
        // CHECK: scop.scats = [0, 0, 0, 0]
        call @S0() : () -> ()
      }
    }
  }

  return
}

func private @S0() attributes {scop.stmt}

// -----

// CHECK-LABEL: func @imperfectly_nested
func @imperfectly_nested(%A: memref<?x?x?xf32>) {
  // expected-remark@above {{Has OslScop: true}}
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index

  %d0 = dim %A, %c0 : memref<?x?x?xf32>
  %d1 = dim %A, %c1 : memref<?x?x?xf32>
  %d2 = dim %A, %c2 : memref<?x?x?xf32>

  affine.for %i = 0 to %d0 {
    // CHECK: scop.scats = [0, 0]
    call @S0() : () -> ()

    affine.for %j = 0 to %d1 {
      affine.for %k = 0 to %d2 {
        // CHECK: scop.scats = [0, 1, 0, 0]
        call @S1() : () -> ()
      }
    }

    // CHECK: scop.scats = [0, 2]
    call @S2() : () -> ()

    affine.for %j = 0 to %d1 {
      // CHECK: scop.scats = [0, 3, 0]
      call @S3() : () -> ()
      // CHECK: scop.scats = [0, 3, 1]
      call @S4() : () -> ()
    }
  }

  // CHECK: scop.scats = [1]
  call @S5() : () -> ()

  return
}

func private @S0() attributes {scop.stmt}
func private @S1() attributes {scop.stmt}
func private @S2() attributes {scop.stmt}
func private @S3() attributes {scop.stmt}
func private @S4() attributes {scop.stmt}
func private @S5() attributes {scop.stmt}
