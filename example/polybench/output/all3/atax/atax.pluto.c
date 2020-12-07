#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

// TODO: mlir-clang %s %stdinclude | FileCheck %s
// RUN: clang %s -O3 %stdinclude %polyverify -o %s.exec1 && %s.exec1 &> %s.out1
// RUN: mlir-clang %s %polyverify %stdinclude -emit-llvm | clang -x ir - -O3 -o %s.execm && %s.execm &> %s.out2
// RUN: rm -f %s.exec1 %s.execm
// RUN: diff %s.out1 %s.out2
// RUN: rm -f %s.out1 %s.out2
// RUN: mlir-clang %s %polyexec %stdinclude -emit-llvm | clang -x ir - -O3 -o %s.execm && %s.execm > %s.mlir.time; cat %s.mlir.time | FileCheck %s --check-prefix EXEC
// RUN: clang %s -O3 %polyexec %stdinclude -o %s.exec2 && %s.exec2 > %s.clang.time; cat %s.clang.time | FileCheck %s --check-prefix EXEC
// RUN: rm -f %s.exec2 %s.execm
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* atax.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "atax.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n))
{
  int i, j;
  DATA_TYPE fn;
  fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
      x[i] = 1 + (i / fn);
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % n) / (5*m);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(y,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
  }
  POLYBENCH_DUMP_END("y");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_atax(int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(tmp,M,m))
{
  int i, j;

/* Copyright (C) 1991-2020 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */
/* This header is separate from features.h so that the compiler can
   include it implicitly at the start of every compilation.  It must
   not itself include <features.h> or any other header that includes
   <features.h> because the implicit include comes before any feature
   test macros that may be defined in a source file before it first
   explicitly includes a system header.  GCC knows the name of this
   header in order to preinclude it.  */
/* glibc's intent is to support the IEC 559 math functionality, real
   and complex.  If the GCC (4.9 and later) predefined macros
   specifying compiler intent are available, use them to determine
   whether the overall intent is to support these features; otherwise,
   presume an older compiler has intent to support these features and
   define these macros by default.  */
/* wchar_t uses Unicode 10.0.0.  Version 10.0 of the Unicode Standard is
   synchronized with ISO/IEC 10646:2017, fifth edition, plus
   the following additions from Amendment 1 to the fifth edition:
   - 56 emoji characters
   - 285 hentaigana
   - 3 additional Zanabazar Square characters */
  int t1, t2, t3, t4, t5;
 register int lbv, ubv;
/* Start of CLooG code */
for (t2=0;t2<=floord(_PB_M-1,32);t2++) {
  lbv=32*t2;
  ubv=min(_PB_M-1,32*t2+31);
#pragma ivdep
#pragma vector always
  for (t3=lbv;t3<=ubv;t3++) {
    tmp[t3] = SCALAR_VAL(0.0);;
  }
}
if (_PB_N >= 1) {
  for (t2=0;t2<=floord(_PB_M-1,32);t2++) {
    for (t3=0;t3<=floord(_PB_N-1,32);t3++) {
      for (t4=32*t2;t4<=min(_PB_M-1,32*t2+31);t4++) {
        for (t5=32*t3;t5<=min(_PB_N-1,32*t3+31);t5++) {
          tmp[t4] = tmp[t4] + A[t4][t5] * x[t5];;
        }
      }
    }
  }
}
for (t2=0;t2<=floord(_PB_N-1,32);t2++) {
  lbv=32*t2;
  ubv=min(_PB_N-1,32*t2+31);
#pragma ivdep
#pragma vector always
  for (t3=lbv;t3<=ubv;t3++) {
    y[t3] = 0;;
  }
}
if (_PB_M >= 1) {
  for (t2=0;t2<=floord(_PB_N-1,32);t2++) {
    for (t3=0;t3<=floord(_PB_M-1,32);t3++) {
      for (t4=32*t3;t4<=min(_PB_M-1,32*t3+31);t4++) {
        lbv=32*t2;
        ubv=min(_PB_N-1,32*t2+31);
#pragma ivdep
#pragma vector always
        for (t5=lbv;t5<=ubv;t5++) {
          y[t5] = y[t5] + A[t4][t5] * tmp[t4];;
        }
      }
    }
  }
}
/* End of CLooG code */

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, N, m, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, M, m);

  /* Initialize array(s). */
  init_array (m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_atax (m, n,
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(x),
	       POLYBENCH_ARRAY(y),
	       POLYBENCH_ARRAY(tmp));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(tmp);

  return 0;
}

// CHECK:   func @kernel_atax(%arg0: i32, %arg1: i32, %arg2: memref<1900x2100xf64>, %arg3: memref<2100xf64>, %arg4: memref<2100xf64>, %arg5: memref<1900xf64>) {
// CHECK-NEXT:  %c0_i32 = constant 0 : i32
// CHECK-NEXT:  %cst = constant 0.000000e+00 : f64
// CHECK-NEXT:  %0 = index_cast %arg1 : i32 to index
// CHECK-NEXT:  %1 = sitofp %c0_i32 : i32 to f64
// CHECK-NEXT:  affine.for %arg6 = 0 to %0 {
// CHECK-NEXT:    affine.store %1, %arg4[%arg6] : memref<2100xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  %2 = index_cast %arg0 : i32 to index
// CHECK-NEXT:  affine.for %arg6 = 0 to %2 {
// CHECK-NEXT:    affine.store %cst, %arg5[%arg6] : memref<1900xf64>
// CHECK-NEXT:    %3 = affine.load %arg5[%arg6] : memref<1900xf64>
// CHECK-NEXT:    affine.for %arg7 = 0 to %0 {
// CHECK-NEXT:      %5 = affine.load %arg2[%arg6, %arg7] : memref<1900x2100xf64>
// CHECK-NEXT:      %6 = affine.load %arg3[%arg7] : memref<2100xf64>
// CHECK-NEXT:      %7 = mulf %5, %6 : f64
// CHECK-NEXT:      %8 = addf %3, %7 : f64
// CHECK-NEXT:      affine.store %8, %arg5[%arg6] : memref<1900xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %4 = affine.load %arg5[%arg6] : memref<1900xf64>
// CHECK-NEXT:    affine.for %arg7 = 0 to %0 {
// CHECK-NEXT:      %5 = affine.load %arg4[%arg7] : memref<2100xf64>
// CHECK-NEXT:      %6 = affine.load %arg2[%arg6, %arg7] : memref<1900x2100xf64>
// CHECK-NEXT:      %7 = mulf %6, %4 : f64
// CHECK-NEXT:      %8 = addf %5, %7 : f64
// CHECK-NEXT:      affine.store %8, %arg4[%arg7] : memref<2100xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}

// EXEC: {{[0-9]\.[0-9]+}}