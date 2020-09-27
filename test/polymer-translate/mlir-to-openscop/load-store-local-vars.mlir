// RUN: polymer-translate %s -mlir-to-openscop | FileCheck %s

// Consider local variables in the domain.
// We will make this test valid when the diff D86421 is landed.

#map = affine_map<(d0) -> (d0 floordiv 2)>

func @load_store_if(%A : memref<?xf32>) -> () {
  %c0 = constant 0 : index
  %N = dim %A, %c0 : memref<?xf32>
  %M = affine.apply #map(%N)

  affine.for %i = 0 to %M {
    %0 = affine.load %A[%i] : memref<?xf32>
    affine.store %0, %A[%i + %M] : memref<?xf32>
  }

  return
}

// CHECK: # [File generated by the OpenScop Library 0.9.2]
// CHECK: <OpenScop>
//
// CHECK: # =============================================== Global
// CHECK: # Language
// CHECK: C
//
// CHECK: # Context
// CHECK: CONTEXT
// CHECK: 1 3 0 0 0 1
// CHECK: # e/i| P0 |  1  
// CHECK:    1    1   -2    ## P0-2 >= 0
//
// CHECK: # Parameters are provided
// CHECK: 1
// CHECK: <strings>
// CHECK: P0
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
// CHECK: 4 5 1 0 1 1
// CHECK: # e/i| i0 | l1 | P0 |  1  
// CHECK:    1    1    0    0    0    ## i0 >= 0
// CHECK:    1    0   -2    1    0    ## -2*l1+P0 >= 0
// CHECK:    1    0    2   -1    1    ## 2*l1-P0+1 >= 0
// CHECK:    1   -1    1    0   -1    ## -i0+l1-1 >= 0
//
// CHECK: # ----------------------------------------------  1.2 Scattering
// CHECK: SCATTERING
// CHECK: 3 8 3 1 1 1
// CHECK: # e/i| c1   c2   c3 | i0 | l1 | P0 |  1  
// CHECK:    0   -1    0    0    0    0    0    0    ## c1 == 0
// CHECK:    0    0   -1    0    1    0    0    0    ## c2 == i0
// CHECK:    0    0    0   -1    0    0    0    0    ## c3 == 0
//
// CHECK: # ----------------------------------------------  1.3 Access
// CHECK: READ
// CHECK: 2 7 2 1 1 1
// CHECK: # e/i| Arr  [1]| i0 | l1 | P0 |  1  
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
// CHECK: 4 5 1 0 1 1
// CHECK: # e/i| i0 | l1 | P0 |  1  
// CHECK:    1    1    0    0    0    ## i0 >= 0
// CHECK:    1    0   -2    1    0    ## -2*l1+P0 >= 0
// CHECK:    1    0    2   -1    1    ## 2*l1-P0+1 >= 0
// CHECK:    1   -1    1    0   -1    ## -i0+l1-1 >= 0
// 
// CHECK: # ----------------------------------------------  2.2 Scattering
// CHECK: SCATTERING
// CHECK: 3 8 3 1 1 1
// CHECK: # e/i| c1   c2   c3 | i0 | l1 | P0 |  1  
// CHECK:    0   -1    0    0    0    0    0    0    ## c1 == 0
// CHECK:    0    0   -1    0    1    0    0    0    ## c2 == i0
// CHECK:    0    0    0   -1    0    0    0    1    ## c3 == 1
// 
// CHECK: # ----------------------------------------------  2.3 Access
// CHECK: WRITE
// CHECK: 2 7 2 1 1 1
// CHECK: # e/i| Arr  [1]| i0 | l1 | P0 |  1  
// CHECK:    0   -1    0    0    0    0    1    ## Arr == A1
// CHECK:    0    0   -1    1    1    0    0    ## [1] == i0+l1
// 
// CHECK: # ----------------------------------------------  2.4 Statement Extensions
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
// CHECK: # =============================================== Extensions
// CHECK: <arrays>
// CHECK: # Number of arrays
// CHECK: 1
// CHECK: # Mapping array-identifiers/array-names
// CHECK: 1 A1
// CHECK: </arrays>
// 
// CHECK: </OpenScop>
