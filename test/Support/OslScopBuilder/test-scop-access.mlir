// RUN: polymer-opt %s -test-osl-scop-builder  -split-input-file | FileCheck %s

// Test how the access constraints are resolved.

// CHECK-LABEL: func @no_callee_body
func @no_callee_body() {
  // expected-remark@above {{Has OslScop: true}}
  call @S0() : () -> ()
  return
}

func private @S0() attributes {scop.stmt}

// -----

// CHECK: #{{.*}} = affine_set<(d0, d1, d2)[s0] : (d1 - d2 == 0, d0 - 1 == 0)>
// CHECK-LABEL: func @identity_address
func @identity_address(%A: memref<?xf32>) {
  // expected-remark@above {{Has OslScop: true}}
  %c0 = constant 0 : index
  %d0 = dim %A, %c0 : memref<?xf32>

  affine.for %i = 0 to %d0 {
    call @S0(%A, %i) : (memref<?xf32>, index) -> ()
  }
  return
}

func private @S0(%A: memref<?xf32>, %i: index) attributes {scop.stmt} {
  %0 = affine.load %A[%i] : memref<?xf32>
  affine.store %0, %A[%i] : memref<?xf32>

  return
}

// -----

// CHECK: #{{.*}} = affine_set<(d0, d1, d2)[s0] : (d1 - d2 == 0, d0 - 1 == 0)>
// CHECK: #{{.*}} = affine_set<(d0, d1, d2)[s0] : (d1 - d2 == 0, d0 - 2 == 0)>
// CHECK-LABEL: func @multi_memrefs
func @multi_memrefs(%A: memref<?xf32>, %B: memref<?xf32>) {
  // expected-remark@above {{Has OslScop: true}}
  %c0 = constant 0 : index
  %d0 = dim %A, %c0 : memref<?xf32>

  affine.for %i = 0 to %d0 {
    call @S0(%A, %B, %i) : (memref<?xf32>, memref<?xf32>, index) -> ()
  }

  return 
}

func private @S0(%A: memref<?xf32>, %B: memref<?xf32>, %i: index) attributes {scop.stmt} {
  %0 = affine.load %A[%i] : memref<?xf32>
  affine.store %0, %B[%i] : memref<?xf32>

  return
}

// -----

// Different domains.

// CHECK: #{{.*}} = affine_set<(d0, d1, d2)[s0] : (d1 - d2 == 0, d0 - 1 == 0)>
// CHECK: #{{.*}} = affine_set<(d0, d1, d2)[s0] : (d1 - d2 == 0, d0 - 2 == 0)>
// CHECK-LABEL: func @multi_domains
func @multi_domains(%A: memref<?xf32>, %B: memref<?xf32>) {
  // expected-remark@above {{Has OslScop: true}}
  %c0 = constant 0 : index
  %d0 = dim %A, %c0 : memref<?xf32>

  affine.for %i = 0 to %d0 {
    call @S0(%A, %B, %i) : (memref<?xf32>, memref<?xf32>, index) -> ()
  }

  affine.for %i = 0 to %d0 {
    call @S1(%A, %B, %i) : (memref<?xf32>, memref<?xf32>, index) -> ()
  }

  return 
}

// CHECK-LABEL: @S0
func private @S0(%A: memref<?xf32>, %B: memref<?xf32>, %i: index) attributes {scop.stmt} {
  // CHECK: scop.access_symbols = ["A1", "", "i1", "P1"]
  %0 = affine.load %A[%i] : memref<?xf32>
  // CHECK: scop.access_symbols = ["A2", "", "i1", "P1"]
  affine.store %0, %B[%i] : memref<?xf32>

  return
}

func private @S1(%A: memref<?xf32>, %B: memref<?xf32>, %i: index) attributes {scop.stmt} {
  // CHECK: scop.access_symbols = ["A1", "", "i2", "P1"]
  %0 = affine.load %A[%i] : memref<?xf32>
  // CHECK: scop.access_symbols = ["A2", "", "i2", "P1"]
  affine.store %0, %B[%i] : memref<?xf32>

  return
}

// -----

// Multi dimensional.

// CHECK: #{{.*}} = affine_set<(d0, d1, d2, d3)[s0, s1] : (d1 - d2 == 0, d0 - 1 == 0)>
// CHECK: #{{.*}} = affine_set<(d0, d1, d2, d3, d4)[s0, s1] : (d1 - d3 == 0, d2 - d4 == 0, d0 - 2 == 0)>
// CHECK-LABEL: @multi_dims
func @multi_dims(%A: memref<?xf32>, %B: memref<?x?xf32>) {
  // expected-remark@above {{Has OslScop: true}}
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  %d0 = dim %B, %c0 : memref<?x?xf32>
  %d1 = dim %B, %c1 : memref<?x?xf32>

  affine.for %i = 0 to %d0 {
    affine.for %j = 0 to %d1 {
      call @S0(%A, %B, %i, %j) : (memref<?xf32>, memref<?x?xf32>, index, index) -> ()
    }
  }

  return 
}

// CHECK-LABEL: @S0
func private @S0(%A: memref<?xf32>, %B: memref<?x?xf32>, %i: index, %j: index) attributes {scop.stmt} {
  // CHECK: scop.access_symbols = ["A1", "", "i1", "i2", "P1", "P2"]
  %0 = affine.load %A[%i] : memref<?xf32>
  // CHECK: scop.access_symbols = ["A2", "", "", "i1", "i2", "P1", "P2"]
  affine.store %0, %B[%i, %j] : memref<?x?xf32>

  return
}
