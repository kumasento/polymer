// RUN: polymer-opt %s -test-osl-scop-builder -verify-diagnostics -split-input-file | FileCheck %s

// Test whether the domain is generated correctly.

// CHECK: #{{.*}} = affine_set<() : (0 == 0)>
// CHECK-LABEL: @no_domain
func @no_domain() {
  // expected-remark@above {{Has OslScop: true}}
  call @S0() : () -> ()
  return
}

func private @S0() attributes {scop.stmt}

// -----

// CHECK: #{{.*}} = affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)> 
// CHECK-LABEL: @single_domain
func @single_domain(%A: memref<?xf32>) {
  // expected-remark@above {{Has OslScop: true}}
  %c0 = constant 0 : index
  %d0 = dim %A, %c0 : memref<?xf32>

  affine.for %i = 0 to %d0 {
    // CHECK: scop.domain_symbols = ["i1", "P1"]
    call @S0() : () -> ()
  }

  return 
}

func private @S0() attributes {scop.stmt}

// -----

// CHECK: #{{.*}} = affine_set<(d0, d1, d2)[s0, s1, s2] : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + s1 - 1 >= 0, d2 >= 0, -d2 + s2 - 1 >= 0)>
// CHECK-LABEL: @multi_domains
func @multi_domains(%A: memref<?x?x?xf32>) {
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
        // CHECK-LABEL: @S0
        // CHECK: scop.domain_symbols = ["i1", "i2", "i3", "P1", "P2", "P3"]
        call @S0() : () -> ()
      }
      // CHECK-LABEL: @S1
      // CHECK: scop.domain_symbols = ["i1", "i2", "P1", "P2", "P3"]
      call @S1() : () -> ()
    }
  }

  // CHECK-LABEL: @S2
  // CHECK: scop.domain_symbols = ["P1", "P2", "P3"]
  call @S2() : () -> ()

  affine.for %i = 0 to %d0 {
    // CHECK-LABEL: @S3
    // CHECK: scop.domain_symbols = ["i4", "P1", "P2", "P3"]
    call @S3() : () -> ()
  }

  return 
}

func private @S0() attributes {scop.stmt}
func private @S1() attributes {scop.stmt}
func private @S2() attributes {scop.stmt}
func private @S3() attributes {scop.stmt}
