// RUN: polymer-translate %s -mlir-to-openscop | FileCheck %s

// Load data from one 1D array to another 2D array.
// This case is used to test the case that two statements that
// have not completely identical parameter sets.

func @load_store_param_2d(%A : memref<?xf32>, %B : memref<?x?xf32>) -> () {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  %M = dim %B, %c0 : memref<?x?xf32>
  %N = dim %B, %c1 : memref<?x?xf32>

  affine.for %i = 0 to %M {
    %0 = affine.load %A[%i] : memref<?xf32>
    affine.for %j = 0 to %N {
      affine.store %0, %B[%i, %j] : memref<?x?xf32>
    }
  }

  return 
}

// # [File generated by the OpenScop Library 0.9.2]
// CHECK: <OpenScop>
//
// CHECK: # =============================================== Global
// CHECK: # Language
// CHECK: C
//
// CHECK: # Context
// CHECK: CONTEXT
// CHECK: 2 4 0 0 0 2
// CHECK: # e/i| P0   P1 |  1  
// CHECK:    1    1    0   -1    ## P0-1 >= 0
// CHECK:    1    0    1   -1    ## P1-1 >= 0
//
// CHECK: # Parameters are provided
// CHECK: 1
// CHECK: <strings>
// CHECK: P0 P1
// CHECK: </strings>
//
// CHECK: # Number of statements
// CHECK: 2
//
// CHECK: # =============================================== Statement 1
// CHECK: # Number of relations describing the statement:
// CHECK: 3
//
// CHECK: # ----------------------------------------------  1.1 Domain
// CHECK: DOMAIN
// CHECK: 2 5 1 0 0 2
// CHECK: # e/i| i0 | P0   P1 |  1  
// CHECK:    1    1    0    0    0    ## i0 >= 0
// CHECK:    1   -1    1    0   -1    ## -i0+P0-1 >= 0
//
// CHECK: # ----------------------------------------------  1.2 Scattering
// CHECK: SCATTERING
// CHECK: 3 8 3 1 0 2
// CHECK: # e/i| c1   c2   c3 | i0 | P0   P1 |  1  
// CHECK:    0   -1    0    0    0    0    0    0    ## c1 == 0
// CHECK:    0    0   -1    0    1    0    0    0    ## c2 == i0
// CHECK:    0    0    0   -1    0    0    0    0    ## c3 == 0
//
// CHECK: # ----------------------------------------------  1.3 Access
// CHECK: READ
// CHECK: 2 7 2 1 0 2
// CHECK: # e/i| Arr  [1]| i0 | P0   P1 |  1  
// CHECK:    0   -1    0    0    0    0    1    ## Arr == A1
// CHECK:    0    0   -1    1    0    0    0    ## [1] == i0
//
// CHECK: # ----------------------------------------------  1.4 Statement Extensions
// CHECK: # Number of Statement Extensions
// CHECK: 1
// CHECK: <body>
// CHECK: # Number of original iterators
// CHECK: 1
// CHECK: # List of original iterators
// CHECK: i0
// CHECK: # Statement body expression
//
// CHECK: </body>
//
// CHECK: # =============================================== Statement 2
// CHECK: # Number of relations describing the statement:
// CHECK: 3
//
// CHECK: # ----------------------------------------------  2.1 Domain
// CHECK: DOMAIN
// CHECK: 4 6 2 0 0 2
// CHECK: # e/i| i0   i1 | P0   P1 |  1  
// CHECK:    1    1    0    0    0    0    ## i0 >= 0
// CHECK:    1   -1    0    1    0   -1    ## -i0+P0-1 >= 0
// CHECK:    1    0    1    0    0    0    ## i1 >= 0
// CHECK:    1    0   -1    0    1   -1    ## -i1+P1-1 >= 0
//
// CHECK: # ----------------------------------------------  2.2 Scattering
// CHECK: SCATTERING
// CHECK: 5 11 5 2 0 2
// CHECK: # e/i| c1   c2   c3   c4   c5 | i0   i1 | P0   P1 |  1  
// CHECK:    0   -1    0    0    0    0    0    0    0    0    0    ## c1 == 0
// CHECK:    0    0   -1    0    0    0    1    0    0    0    0    ## c2 == i0
// CHECK:    0    0    0   -1    0    0    0    0    0    0    1    ## c3 == 1
// CHECK:    0    0    0    0   -1    0    0    1    0    0    0    ## c4 == i1
// CHECK:    0    0    0    0    0   -1    0    0    0    0    0    ## c5 == 0
//
// CHECK: # ----------------------------------------------  2.3 Access
// CHECK: WRITE
// CHECK: 3 9 3 2 0 2
// CHECK: # e/i| Arr  [1]  [2]| i0   i1 | P0   P1 |  1  
// CHECK:    0   -1    0    0    0    0    0    0    2    ## Arr == A2
// CHECK:    0    0   -1    0    1    0    0    0    0    ## [1] == i0
// CHECK:    0    0    0   -1    0    1    0    0    0    ## [2] == i1
//
// CHECK: # ----------------------------------------------  2.4 Statement Extensions
// CHECK: # Number of Statement Extensions
// CHECK: 1
// CHECK: <body>
// CHECK: # Number of original iterators
// CHECK: 2
// CHECK: # List of original iterators
// CHECK: i0 i1
// CHECK: # Statement body expression
//
// CHECK: </body>
//
// CHECK: # =============================================== Extensions
// CHECK: <arrays>
// CHECK: # Number of arrays
// CHECK: 2
// CHECK: # Mapping array-identifiers/array-names
// CHECK: </arrays>
//
// CHECK: </OpenScop>
