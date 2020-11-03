; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare i8* @malloc(i64)

declare void @free(i8*)

define void @load_store_2d(float* %0, float* %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, float* %7, float* %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13) !dbg !3 {
  %15 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } undef, float* %0, 0, !dbg !7
  %16 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %15, float* %1, 1, !dbg !9
  %17 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %16, i64 %2, 2, !dbg !10
  %18 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %17, i64 %3, 3, 0, !dbg !11
  %19 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %18, i64 %5, 4, 0, !dbg !12
  %20 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %19, i64 %4, 3, 1, !dbg !13
  %21 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %20, i64 %6, 4, 1, !dbg !14
  %22 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } undef, float* %7, 0, !dbg !15
  %23 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %22, float* %8, 1, !dbg !16
  %24 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %23, i64 %9, 2, !dbg !17
  %25 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %24, i64 %10, 3, 0, !dbg !18
  %26 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %25, i64 %12, 4, 0, !dbg !19
  %27 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %26, i64 %11, 3, 1, !dbg !20
  %28 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %27, i64 %13, 4, 1, !dbg !21
  %29 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %21, 3, 0, !dbg !22
  %30 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %21, 3, 1, !dbg !23
  br label %31, !dbg !24

31:                                               ; preds = %55, %14
  %32 = phi i64 [ %56, %55 ], [ 0, %14 ]
  %33 = icmp slt i64 %32, %29, !dbg !25
  br i1 %33, label %34, label %57, !dbg !26

34:                                               ; preds = %31
  br label %35, !dbg !27

35:                                               ; preds = %38, %34
  %36 = phi i64 [ %54, %38 ], [ 0, %34 ]
  %37 = icmp slt i64 %36, %30, !dbg !28
  br i1 %37, label %38, label %55, !dbg !29

38:                                               ; preds = %35
  %39 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %21, 1, !dbg !30
  %40 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %21, 4, 0, !dbg !31
  %41 = mul i64 %32, %40, !dbg !32
  %42 = add i64 0, %41, !dbg !33
  %43 = mul i64 %36, 1, !dbg !34
  %44 = add i64 %42, %43, !dbg !35
  %45 = getelementptr float, float* %39, i64 %44, !dbg !36
  %46 = load float, float* %45, align 4, !dbg !37
  %47 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %28, 1, !dbg !38
  %48 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %28, 4, 0, !dbg !39
  %49 = mul i64 %32, %48, !dbg !40
  %50 = add i64 0, %49, !dbg !41
  %51 = mul i64 %36, 1, !dbg !42
  %52 = add i64 %50, %51, !dbg !43
  %53 = getelementptr float, float* %47, i64 %52, !dbg !44
  store float %46, float* %53, align 4, !dbg !45
  %54 = add i64 %36, 1, !dbg !46
  br label %35, !dbg !47

55:                                               ; preds = %35
  %56 = add i64 %32, 1, !dbg !48
  br label %31, !dbg !49

57:                                               ; preds = %31
  ret void, !dbg !50
}

define void @_mlir_ciface_load_store_2d({ float*, float*, i64, [2 x i64], [2 x i64] }* %0, { float*, float*, i64, [2 x i64], [2 x i64] }* %1) !dbg !51 {
  %3 = load { float*, float*, i64, [2 x i64], [2 x i64] }, { float*, float*, i64, [2 x i64], [2 x i64] }* %0, align 8, !dbg !52
  %4 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %3, 0, !dbg !54
  %5 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %3, 1, !dbg !55
  %6 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %3, 2, !dbg !56
  %7 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %3, 3, 0, !dbg !57
  %8 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %3, 3, 1, !dbg !58
  %9 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %3, 4, 0, !dbg !59
  %10 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %3, 4, 1, !dbg !60
  %11 = load { float*, float*, i64, [2 x i64], [2 x i64] }, { float*, float*, i64, [2 x i64], [2 x i64] }* %1, align 8, !dbg !61
  %12 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %11, 0, !dbg !62
  %13 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %11, 1, !dbg !63
  %14 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %11, 2, !dbg !64
  %15 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %11, 3, 0, !dbg !65
  %16 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %11, 3, 1, !dbg !66
  %17 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %11, 4, 0, !dbg !67
  %18 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %11, 4, 1, !dbg !68
  call void @load_store_2d(float* %4, float* %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, float* %12, float* %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18), !dbg !69
  ret void, !dbg !70
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "load_store_2d", linkageName: "load_store_2d", scope: null, file: !4, line: 2, type: !5, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "main_llvm.mlir", directory: "/mnt/ccnas2/bdp/rz3515/projects/polymer/build/test/integration/load-store-2d")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 4, column: 10, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!9 = !DILocation(line: 5, column: 10, scope: !8)
!10 = !DILocation(line: 6, column: 10, scope: !8)
!11 = !DILocation(line: 7, column: 10, scope: !8)
!12 = !DILocation(line: 8, column: 10, scope: !8)
!13 = !DILocation(line: 9, column: 10, scope: !8)
!14 = !DILocation(line: 10, column: 10, scope: !8)
!15 = !DILocation(line: 12, column: 10, scope: !8)
!16 = !DILocation(line: 13, column: 11, scope: !8)
!17 = !DILocation(line: 14, column: 11, scope: !8)
!18 = !DILocation(line: 15, column: 11, scope: !8)
!19 = !DILocation(line: 16, column: 11, scope: !8)
!20 = !DILocation(line: 17, column: 11, scope: !8)
!21 = !DILocation(line: 18, column: 11, scope: !8)
!22 = !DILocation(line: 21, column: 11, scope: !8)
!23 = !DILocation(line: 22, column: 11, scope: !8)
!24 = !DILocation(line: 25, column: 5, scope: !8)
!25 = !DILocation(line: 27, column: 11, scope: !8)
!26 = !DILocation(line: 28, column: 5, scope: !8)
!27 = !DILocation(line: 32, column: 5, scope: !8)
!28 = !DILocation(line: 34, column: 11, scope: !8)
!29 = !DILocation(line: 35, column: 5, scope: !8)
!30 = !DILocation(line: 37, column: 11, scope: !8)
!31 = !DILocation(line: 39, column: 11, scope: !8)
!32 = !DILocation(line: 40, column: 11, scope: !8)
!33 = !DILocation(line: 41, column: 11, scope: !8)
!34 = !DILocation(line: 43, column: 11, scope: !8)
!35 = !DILocation(line: 44, column: 11, scope: !8)
!36 = !DILocation(line: 45, column: 11, scope: !8)
!37 = !DILocation(line: 46, column: 11, scope: !8)
!38 = !DILocation(line: 47, column: 11, scope: !8)
!39 = !DILocation(line: 49, column: 11, scope: !8)
!40 = !DILocation(line: 50, column: 11, scope: !8)
!41 = !DILocation(line: 51, column: 11, scope: !8)
!42 = !DILocation(line: 53, column: 11, scope: !8)
!43 = !DILocation(line: 54, column: 11, scope: !8)
!44 = !DILocation(line: 55, column: 11, scope: !8)
!45 = !DILocation(line: 56, column: 5, scope: !8)
!46 = !DILocation(line: 57, column: 11, scope: !8)
!47 = !DILocation(line: 58, column: 5, scope: !8)
!48 = !DILocation(line: 60, column: 11, scope: !8)
!49 = !DILocation(line: 61, column: 5, scope: !8)
!50 = !DILocation(line: 63, column: 5, scope: !8)
!51 = distinct !DISubprogram(name: "_mlir_ciface_load_store_2d", linkageName: "_mlir_ciface_load_store_2d", scope: null, file: !4, line: 65, type: !5, scopeLine: 65, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!52 = !DILocation(line: 66, column: 10, scope: !53)
!53 = !DILexicalBlockFile(scope: !51, file: !4, discriminator: 0)
!54 = !DILocation(line: 67, column: 10, scope: !53)
!55 = !DILocation(line: 68, column: 10, scope: !53)
!56 = !DILocation(line: 69, column: 10, scope: !53)
!57 = !DILocation(line: 70, column: 10, scope: !53)
!58 = !DILocation(line: 71, column: 10, scope: !53)
!59 = !DILocation(line: 72, column: 10, scope: !53)
!60 = !DILocation(line: 73, column: 10, scope: !53)
!61 = !DILocation(line: 74, column: 10, scope: !53)
!62 = !DILocation(line: 75, column: 10, scope: !53)
!63 = !DILocation(line: 76, column: 11, scope: !53)
!64 = !DILocation(line: 77, column: 11, scope: !53)
!65 = !DILocation(line: 78, column: 11, scope: !53)
!66 = !DILocation(line: 79, column: 11, scope: !53)
!67 = !DILocation(line: 80, column: 11, scope: !53)
!68 = !DILocation(line: 81, column: 11, scope: !53)
!69 = !DILocation(line: 82, column: 5, scope: !53)
!70 = !DILocation(line: 83, column: 5, scope: !53)
