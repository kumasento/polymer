// RUN: polymer-opt %s -test-osl-scop-builder -verify-diagnostics -split-input-file | FileCheck %s

// Test whether the scop.iv_name attribute has been attached.

// CHECK-LABEL: func @test_single_iv()
func @test_single_iv() {
  // expected-remark@above {{Has OslScop: true}}
  affine.for %i = 0 to 10 {
    call @S0() : () -> ()
  }
  // CHECK: {scop.iv_name = "i1"}
  return
}

func private @S0() attributes {scop.stmt}

// -----

// We don't really care about the ordering of the IVs for now.

// CHECK-LABEL: func @test_nested_ivs()
func @test_nested_ivs() {
  // expected-remark@above {{Has OslScop: true}}
  affine.for %i = 0 to 10 {
    affine.for %j = 0 to 10 {
      call @S0() : () -> ()
    }
    // CHECK: {scop.iv_name = {{.*}}}
    call @S1() : () -> ()
  }
  // CHECK: {scop.iv_name = {{.*}}}
  affine.for %i = 0 to 20 {
    call @S2() : () -> ()
  }
  // CHECK: {scop.iv_name = {{.*}}}
  return
}

func private @S0() attributes {scop.stmt}
func private @S1() attributes {scop.stmt}
func private @S2() attributes {scop.stmt}
