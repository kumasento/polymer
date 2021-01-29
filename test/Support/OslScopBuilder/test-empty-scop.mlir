// RUN: polymer-opt %s -test-osl-scop-builder -verify-diagnostics | FileCheck %s

// Test when there is no OslScop can be built.

// CHECK-LABEL: func @test_empty_scop()
func @test_empty_scop() {
  // expected-remark@above {{Has OslScop: false}}
  return
}
