// RUN: polymer-opt %s -test-osl-scop-builder -verify-diagnostics -split-input-file | FileCheck %s

// Test whether parameters can be marked as symbols.

// CHECK-LABEL: func @params_in_top_level
// CHECK: scop.arg_names = ["P1", "P2"]
// CHECK: scop.ctx_params = ["P1", "P2"]
func @params_in_top_level(%N: index, %M: index) {
  // expected-remark@above {{Has OslScop: true}}
  affine.for %i = 0 to %N {
    affine.for %j = 0 to %M {
      call @S0(): () -> ()
    }
  }

  return
}

// CHECK-LABEL: func private @S0()
func private @S0() attributes { scop.stmt }

// -----

// Constant will be folded in this case.

// CHECK-LABEL: func @params_in_body_folded
func @params_in_body_folded() {
  // expected-remark@above {{Has OslScop: true}}
  %N = constant 10 : index

  affine.for %i = 0 to %N {
    call @S0(): () -> ()
  }

  return
}

// CHECK-LABEL: func private @S0()
func private @S0() attributes { scop.stmt }

// -----

// CHECK-LABEL: func @params_in_body
// CHECK: scop.ctx_params = ["P1"]
func @params_in_body(%A: memref<?xf32>) {
  // expected-remark@above {{Has OslScop: true}}
  %c0 = constant 0 : index
  %s0 = dim %A, %c0 : memref<?xf32>
  // CHECK: %[[VAL0:.*]] = dim {scop.param_names = ["P1"]} %{{.*}}, %{{.*}} : memref<?xf32>

  affine.for %i = 0 to %s0 {
    call @S0(): () -> ()
  }

  return 
}

// CHECK-LABEL: func private @S0()
func private @S0() attributes { scop.stmt }

// -----

// Index cast won't print out the attributes.
// AffineApplyNormalizer seems not working.

#map0 = affine_map<()[s0, s1] -> (s0, s1)>

// CHECK-LABEL: func @params_through_affine_map
// CHECK: scop.ctx_params = ["P1", "P2"]
func @params_through_affine_map(%A: memref<?xf32>, %N: i32) {
  // expected-remark@above {{Has OslScop: true}}
  %c0 = constant 0 : index
  %s0 = dim %A, %c0 : memref<?xf32>
  // CHECK: %[[VAL0:.*]] = dim {scop.param_names = ["P1"]} %{{.*}}, %{{.*}} : memref<?xf32>
  %s1 = index_cast %N : i32 to index
  // CHECK: %[[VAL1:.*]] = index_cast %{{.*}} : i32 to index

  affine.for %i = 0 to min #map0()[%s0, %s1] {
    call @S0(): () -> ()
  }

  return 
}

// CHECK-LABEL: func private @S0()
func private @S0() attributes { scop.stmt }
