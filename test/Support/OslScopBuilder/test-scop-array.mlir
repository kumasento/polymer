// RUN: polymer-opt %s -test-osl-scop-builder -verify-diagnostics -split-input-file | FileCheck %s

// Test whether arrays can be extracted and annotated with symbols.

// CHECK-LABEL: func @test_single_array_in_args
// CHECK: scop.arg_names = ["A1"]
func @test_single_array_in_args(%A: memref<10xf32>) {
  // expected-remark@above {{Has OslScop: true}}
  affine.for %i =0 to 10 {
    call @S0(%A) : (memref<10xf32>) -> ()
  }
  return
}

func private @S0(%A: memref<10xf32>) attributes {scop.stmt}
