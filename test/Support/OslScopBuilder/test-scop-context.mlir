// RUN: polymer-opt %s -test-osl-scop-builder -verify-diagnostics -split-input-file | FileCheck %s

// CHECK: #{{.*}} = affine_set<() : (0 == 0)>
// CHECK-LABEL: func @test_empty_context()
func @test_empty_context() {
  // expected-remark@above {{Has OslScop: true}}
  affine.for %i = 0 to 10 {
    call @S0() : () -> ()
  }
  return
}

// CHECK-LABEL: func @S0()
func @S0() attributes { scop.stmt } {
  return
}

// -----

// CHECK: #{{.*}} = affine_set<()[s0] : (s0 - 1 >= 0)>
// CHECK: func @test_ctx_wp
func @test_ctx_wp(%N: index) {
  // expected-remark@above {{Has OslScop: true}}
  affine.for %i = 0 to %N {
    call @S0() : () -> ()
  }
  return
}

// CHECK-LABEL: func @S0()
func @S0() attributes { scop.stmt } {
  return
}

// -----

// CHECK: #[[SET:.*]] = affine_set<()[s0, s1, s2] : (s1 - 1 >= 0, s0 - 1 >= 0, s2 - 1 >= 0)>
// CHECK: func @test_ctx_multi_domains({{.*}}) attributes {
// CHECK-SAME: scop.arg_names = ["P0", "P1", "P2"],
// CHECK-SAME: scop.ctx = #[[SET]],
// CHECK-SAME: scop.ctx_params = ["P0", "P1", "P2"]}
func @test_ctx_multi_domains(%N: index, %M: index, %K: index) {
  // expected-remark@above {{Has OslScop: true}}
  affine.for %i = 0 to %N {
    call @S0() : () -> ()
  }
  affine.for %i = 0 to %M {
    affine.for %j = 0 to %N {
      call @S1() : () -> ()
    }
  }
  affine.for %i = 0 to %K {
    affine.for %j = 0 to %N {
      affine.for %k = 0 to %M {
        call @S2() : () -> ()
      }
    }
  }
  return
}

// CHECK-LABEL: func private @S0()
func private @S0() attributes { scop.stmt }
// CHECK-LABEL: func private @S1()
func private @S1() attributes { scop.stmt }
// CHECK-LABEL: func private @S2()
func private @S2() attributes { scop.stmt }
