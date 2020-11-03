module attributes {llvm.data_layout = ""} {
  llvm.func @load_store_2d(%arg0: !llvm.ptr<float>, %arg1: !llvm.ptr<float>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.ptr<float>, %arg8: !llvm.ptr<float>, %arg9: !llvm.i64, %arg10: !llvm.i64, %arg11: !llvm.i64, %arg12: !llvm.i64, %arg13: !llvm.i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.insertvalue %arg8, %9[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.insertvalue %arg9, %10[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.insertvalue %arg10, %11[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %arg12, %12[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %arg11, %13[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %arg13, %14[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.mlir.constant(0 : index) : !llvm.i64
    %17 = llvm.mlir.constant(1 : index) : !llvm.i64
    %18 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.mlir.constant(0 : index) : !llvm.i64
    %21 = llvm.mlir.constant(1 : index) : !llvm.i64
    llvm.br ^bb1(%20 : !llvm.i64)
  ^bb1(%22: !llvm.i64):  // 2 preds: ^bb0, ^bb5
    %23 = llvm.icmp "slt" %22, %18 : !llvm.i64
    llvm.cond_br %23, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %24 = llvm.mlir.constant(0 : index) : !llvm.i64
    %25 = llvm.mlir.constant(1 : index) : !llvm.i64
    llvm.br ^bb3(%24 : !llvm.i64)
  ^bb3(%26: !llvm.i64):  // 2 preds: ^bb2, ^bb4
    %27 = llvm.icmp "slt" %26, %19 : !llvm.i64
    llvm.cond_br %27, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %28 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %29 = llvm.mlir.constant(0 : index) : !llvm.i64
    %30 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.mul %22, %30 : !llvm.i64
    %32 = llvm.add %29, %31 : !llvm.i64
    %33 = llvm.mlir.constant(1 : index) : !llvm.i64
    %34 = llvm.mul %26, %33 : !llvm.i64
    %35 = llvm.add %32, %34 : !llvm.i64
    %36 = llvm.getelementptr %28[%35] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %37 = llvm.load %36 : !llvm.ptr<float>
    %38 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %39 = llvm.mlir.constant(0 : index) : !llvm.i64
    %40 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %41 = llvm.mul %22, %40 : !llvm.i64
    %42 = llvm.add %39, %41 : !llvm.i64
    %43 = llvm.mlir.constant(1 : index) : !llvm.i64
    %44 = llvm.mul %26, %43 : !llvm.i64
    %45 = llvm.add %42, %44 : !llvm.i64
    %46 = llvm.getelementptr %38[%45] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    llvm.store %37, %46 : !llvm.ptr<float>
    %47 = llvm.add %26, %25 : !llvm.i64
    llvm.br ^bb3(%47 : !llvm.i64)
  ^bb5:  // pred: ^bb3
    %48 = llvm.add %22, %21 : !llvm.i64
    llvm.br ^bb1(%48 : !llvm.i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
  llvm.func @_mlir_ciface_load_store_2d(%arg0: !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>) {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.extractvalue %8[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.extractvalue %8[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @load_store_2d(%1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15) : (!llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
}
