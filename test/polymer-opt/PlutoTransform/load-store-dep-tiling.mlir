// RUN: polymer-opt %s -pluto-opt | FileCheck %s

func @load_store_dep_tiling() {
  %A = alloc() : memref<64x64xf32>

  affine.for %i0 = 1 to 64 {
    affine.for %j0 = 1 to 64 {
      %i1 = affine.apply affine_map<(d0) -> (d0 - 1)>(%i0)
      %j1 = affine.apply affine_map<(d0) -> (d0 - 1)>(%j0)

      %0 = affine.load %A[%i0, %j1] : memref<64x64xf32>
      %1 = affine.load %A[%i1, %j0] : memref<64x64xf32>
      %2 = addf %0, %1 : f32
      affine.store %2, %A[%i0, %j0] : memref<64x64xf32>
    }
  }

  return
}
