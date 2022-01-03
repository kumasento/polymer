# Polymer: bridging polyhedral tools to MLIR

[![Build and Test](https://github.com/kumasento/polymer/actions/workflows/buildAndTest.yml/badge.svg)](https://github.com/kumasento/polymer/actions/workflows/buildAndTest.yml)
[![wakatime](https://wakatime.com/badge/github/kumasento/polymer.svg)](https://wakatime.com/badge/github/kumasento/polymer)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/kumasento/polymer)
![GitHub](https://img.shields.io/github/license/kumasento/polymer)
![GitHub issues](https://img.shields.io/github/issues/kumasento/polymer)
![GitHub pull requests](https://img.shields.io/github/issues-pr/kumasento/polymer)

Bridging polyhedral analysis tools to the MLIR framework.

Polymer is a component of the [Polygeist](https://github.com/wsmoses/Polygeist) framework.
Please read on to find [how to install](#install-polymer) and [use](#basic-usage) Polymer.

## Related Publications/Talks

[[bibtex](resources/polymer.bib)]

### Papers

Polymer is a essential component to the following two papers:

* [Polygeist: Affine C in MLIR (IMPACT'21)](https://acohen.gitlabpages.inria.fr/impact/impact2021/papers/IMPACT_2021_paper_1.pdf). This paper gives an overview of the whole Polygeist framework, in which Polymer does the polyhedral optimisation part of work.
* [Phism: Polyhedral HLS in MLIR (LATTE'21)](https://capra.cs.cornell.edu/latte21/paper/1.pdf). This paper demonstrates an interesting way to leverage Polymer for polyhedral HLS within the MLIR ecosystem.
* [Polygeist: Raising C to Polyhedral MLIR (PACT'21)](https://c.wsmoses.com/papers/Polygeist_PACT.pdf). This is an updated version to the IMPACT'21 submission.

### Talks

Polymer appears in the following talks:

* MLIR Open Dsign Meeting (11/02/2021) on Polygeist. [[slides](https://drive.google.com/file/d/1YJhPBpW77WX53Rxxt2TLbEhdbrOFwDy4/view?usp=sharing)] [[recording](https://drive.google.com/file/d/1P14UrXMlR6WbHR_YrSJVsb7h3cLdr5-h/view?usp=sharing)]
* [LATTE '21](https://capra.cs.cornell.edu/latte21/) on Phism. [[recording](https://youtu.be/50UjVlDF1Us)]


## Install Polymer

[[legacy installation method](docs/LEGACY_INSTALL_METHOD.md)]
[[submodule problem](docs/WHY_NOT_SUBMODULE_LLVM.md)]

The recommended way of installing Polymer is to have it as a component of [Polygeist](https://github.com/wsmoses/Polygeist). Please find the detailed instruction [here](docs/INSTALL_WITHIN_POLYGEIST.md).

You can also install Polymer as an [individual, out-of-tree project](docs/INSTALL_INDIVIDUALLY.md). 

## Basic usage

Optimize MLIR code described in the Affine dialect by Pluto:

```mlir
// File name: matmul.mlir
func @matmul() {
  %A = memref.alloc() : memref<64x64xf32>
  %B = memref.alloc() : memref<64x64xf32>
  %C = memref.alloc() : memref<64x64xf32>

  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      affine.for %k = 0 to 64 {
        %0 = affine.load %A[%i, %k] : memref<64x64xf32>
        %1 = affine.load %B[%k, %j] : memref<64x64xf32>
        %2 = arith.mulf %0, %1 : f32
        %3 = affine.load %C[%i, %j] : memref<64x64xf32>
        %4 = arith.addf %2, %3 : f32
        affine.store %4, %C[%i, %j] : memref<64x64xf32>
      }
    }
  }

  return
}
```

The following command will optimize this code piece.

```shell
# Go to the build/ directory.
./bin/polymer-opt -reg2mem -extract-scop-stmt -pluto-opt matmul.mlir 
```

Output:

```mlir
#map0 = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0) -> (d0 * 32 + 32)>
module  {
  func private @S0(%arg0: index, %arg1: index, %arg2: memref<64x64xf32>, %arg3: index, %arg4: memref<64x64xf32>, %arg5: memref<64x64xf32>) attributes {scop.stmt} {
    %0 = affine.load %arg5[symbol(%arg0), symbol(%arg3)] : memref<64x64xf32>
    %1 = affine.load %arg4[symbol(%arg3), symbol(%arg1)] : memref<64x64xf32>
    %2 = arith.mulf %0, %1 : f32
    %3 = affine.load %arg2[symbol(%arg0), symbol(%arg1)] : memref<64x64xf32>
    %4 = arith.addf %2, %3 : f32
    affine.store %4, %arg2[symbol(%arg0), symbol(%arg1)] : memref<64x64xf32>
    return
  }
  
  func @matmul() {
    %0 = memref.alloc() : memref<64x64xf32>
    %1 = memref.alloc() : memref<64x64xf32>
    %2 = memref.alloc() : memref<64x64xf32>
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        affine.for %arg2 = 0 to 2 {
          affine.for %arg3 = #map0(%arg0) to #map1(%arg0) {
            affine.for %arg4 = #map0(%arg2) to #map1(%arg2) {
              affine.for %arg5 = #map0(%arg1) to #map1(%arg1) {
                call @S0(%arg3, %arg5, %0, %arg4, %1, %2) : (index, index, memref<64x64xf32>, index, memref<64x64xf32>, memref<64x64xf32>) -> ()
              }
            }
          }
        }
      }
    }
    return
  }
}
```
