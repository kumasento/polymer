// RUN: polymer-opt %s -test-scat-tree -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @no_affine_for()
func @no_affine_for() -> i32 {
  %c0 = constant 0 : i32
  // expected-remark@above {{Scats: { 0 }}}
  return %c0: i32
  // expected-remark@above {{Scats: { 1 }}}
}

// -----

// CHECK-LABEL: func @single_affine_for
func @single_affine_for() {
  %A = alloc() : memref<10xf32>
  // expected-remark@above {{Scats: { 0 }}}
  affine.for %i = 0 to 10 {
    %0 = affine.load %A[%i] : memref<10xf32>
    // expected-remark@above {{Scats: { 1, 0 }}}
    %1 = mulf %0, %0 : f32
    // expected-remark@above {{Scats: { 1, 1 }}}
    affine.store %1, %A[%i] : memref<10xf32>
    // expected-remark@above {{Scats: { 1, 2 }}}
  }
  return
  // expected-remark@above {{Scats: { 2 }}}
}

// -----

// CHECK-LABEL: func @nested_affine_for
func @nested_affine_for() {
  %A = alloc() : memref<64x64xf32>
  // expected-remark@above {{Scats: { 0 }}}
  %B = alloc() : memref<64x64xf32>
  // expected-remark@above {{Scats: { 1 }}}
  %C = alloc() : memref<64x64xf32>
  // expected-remark@above {{Scats: { 2 }}}

  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      affine.for %k = 0 to 64 {
        %0 = affine.load %A[%i, %k] : memref<64x64xf32>
        // expected-remark@above {{Scats: { 3, 0, 0, 0 }}}
        %1 = affine.load %B[%k, %j] : memref<64x64xf32>
        // expected-remark@above {{Scats: { 3, 0, 0, 1 }}}
        %2 = mulf %0, %1 : f32
        // expected-remark@above {{Scats: { 3, 0, 0, 2 }}}
        %3 = affine.load %C[%i, %j] : memref<64x64xf32>
        // expected-remark@above {{Scats: { 3, 0, 0, 3 }}}
        %4 = addf %2, %3 : f32
        // expected-remark@above {{Scats: { 3, 0, 0, 4 }}}
        affine.store %4, %C[%i, %j] : memref<64x64xf32>
        // expected-remark@above {{Scats: { 3, 0, 0, 5 }}}
      }
    }
  }
  return
  // expected-remark@above {{Scats: { 4 }}}
}

// -----

// CHECK-LABEL: func @imperfectly_nested_affine_for
func @imperfectly_nested_affine_for(%alpha: f32) {
  %A = alloc() : memref<64x64xf32>
  // expected-remark@above {{Scats: { 0 }}}
  %B = alloc() : memref<64x64xf32>
  // expected-remark@above {{Scats: { 1 }}}
  %C = alloc() : memref<64x64xf32>
  // expected-remark@above {{Scats: { 2 }}}

  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      %0 = affine.load %C[%i, %j] : memref<64x64xf32>
      // expected-remark@above {{Scats: { 3, 0, 0 }}}
      %1 = mulf %alpha, %0 : f32
      // expected-remark@above {{Scats: { 3, 0, 1 }}}
      affine.store %1, %C[%i, %j] : memref<64x64xf32>
      // expected-remark@above {{Scats: { 3, 0, 2 }}}

      affine.for %k = 0 to 64 {
        %2 = affine.load %A[%i, %k] : memref<64x64xf32>
        // expected-remark@above {{Scats: { 3, 0, 3, 0 }}}
        %3 = affine.load %B[%k, %j] : memref<64x64xf32>
        // expected-remark@above {{Scats: { 3, 0, 3, 1 }}}
        %4 = mulf %2, %3 : f32
        // expected-remark@above {{Scats: { 3, 0, 3, 2 }}}
        %5 = affine.load %C[%i, %j] : memref<64x64xf32>
        // expected-remark@above {{Scats: { 3, 0, 3, 3 }}}
        %6 = addf %4, %5 : f32
        // expected-remark@above {{Scats: { 3, 0, 3, 4 }}}
        affine.store %6, %C[%i, %j] : memref<64x64xf32>
        // expected-remark@above {{Scats: { 3, 0, 3, 5 }}}
      }
    }
  }
  return
  // expected-remark@above {{Scats: { 4 }}}
}
