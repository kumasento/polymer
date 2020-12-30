// RUN: polymer-opt %s -test-osl-scop-builder -verify-diagnostics -split-input-file | FileCheck %s

// CHECK-LABEL: func @test_empty_context()
func @test_empty_context() {
  // expected-remark@above {{Has OslScop: true}}
  // expected-remark@above {{Num context parameters: 0}}
  // expected-remark@above {{eqs: {} inEqs: {}}}
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

// CHECK-LABEL: func @test_context_with_parameters
func @test_context_with_parameters(%N: index) {
  // expected-remark@above {{Has OslScop: true}}
  // expected-remark@above {{Num context parameters: 1}}
  // expected-remark@above {{eqs: {} inEqs: {{ 1, -1 }}}}
  affine.for %i = 0 to %N {
    call @S0() : () -> ()
  }
  return
}

// CHECK-LABEL: func @S0()
func @S0() attributes { scop.stmt } {
  return
}
