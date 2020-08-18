// RUN: emit-openscop %s -emit-openscop | FileCheck %s

func @load_store() -> () {
  %A = alloc() : memref<32xf32>
  affine.for %i = 0 to 32 {
    %0 = affine.load %A[%i] : memref<32xf32>
    affine.store %0, %A[%i] : memref<32xf32>
  }
  return
}

// CHECK: <OpenScop>
// CHECK: # =============================================== Global
// CHECK: # Backend Language
// CHECK: C
// CHECK: # Context
// CHECK: CONTEXT
// CHECK: 0 2 0 0 0 0
// CHECK: # Parameter names are provided
// CHECK: 0
// CHECK: # Number of statements
// CHECK: 2
// CHECK: # =============================================== Statement 1
// CHECK: # Number of relations describing the statement
// CHECK: 3
// CHECK: # ----------------------------------------------  1.1 Domain
// CHECK: DOMAIN
// CHECK: 2 3 1 0 0 0
// CHECK: # e/i | i0 | | 1
// CHECK:   1   1   0 
// CHECK:   1  -1  31 
// CHECK: # ----------------------------------------------  1.2 Scattering
// CHECK: SCATTERING
// CHECK: 3 6 3 1 0 0
// CHECK: # e/i | s0 s1 s2 | i0 | | 1
// CHECK:   0  -1   0   0   0   0 
// CHECK:   0   0  -1   0   1   0 
// CHECK:   0   0   0  -1   0   0 
// CHECK: # ----------------------------------------------  1.3 Access
// CHECK: READ
// CHECK: 2 5 2 1 0 0
// CHECK: # e/i | Arr [i0] | i0 | | 1
// CHECK:   0  -1   0   0   1 
// CHECK:   0   0  -1   1   0   0 
// CHECK: # ----------------------------------------------  Statement Extensions
// CHECK: # Number of Statement Extensions
// CHECK: 0
// CHECK: # =============================================== Statement 2
// CHECK: # Number of relations describing the statement
// CHECK: 3
// CHECK: # ----------------------------------------------  2.1 Domain
// CHECK: DOMAIN
// CHECK: 2 3 1 0 0 0
// CHECK: # e/i | i0 | | 1
// CHECK:   1   1   0 
// CHECK:   1  -1  31 
// CHECK: # ----------------------------------------------  2.2 Scattering
// CHECK: SCATTERING
// CHECK: 3 6 3 1 0 0
// CHECK: # e/i | s0 s1 s2 | i0 | | 1
// CHECK:   0  -1   0   0   0   0 
// CHECK:   0   0  -1   0   1   0 
// CHECK:   0   0   0  -1   0   1 
// CHECK: # ----------------------------------------------  2.3 Access
// CHECK: WRITE
// CHECK: 2 5 2 1 0 0
// CHECK: # e/i | Arr [i0] | i0 | | 1
// CHECK:   0  -1   0   0   1 
// CHECK:   0   0  -1   1   0   0 
// CHECK: # ----------------------------------------------  Statement Extensions
// CHECK: # Number of Statement Extensions
// CHECK: 0
// CHECK: </OpenScop>