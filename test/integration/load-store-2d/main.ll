; ModuleID = 'main.c'
source_filename = "main.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.TwoDMemrefF32 = type { float*, float*, i64, [2 x i64], [2 x i64] }

@.str = private unnamed_addr constant [7 x i8] c"%8.6f \00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main(i32 %argc, i8** %argv) #0 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %A = alloca [6 x [8 x float]], align 16
  %B = alloca [6 x [8 x float]], align 16
  %A_mem = alloca %struct.TwoDMemrefF32, align 8
  %B_mem = alloca %struct.TwoDMemrefF32, align 8
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  store i8** %argv, i8*** %argv.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc20, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 6
  br i1 %cmp, label %for.body, label %for.end22

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %j, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %1 = load i32, i32* %j, align 4
  %cmp2 = icmp slt i32 %1, 8
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %2 = load i32, i32* %i, align 4
  %conv = sitofp i32 %2 to float
  %3 = load i32, i32* %j, align 4
  %conv4 = sitofp i32 %3 to float
  %add = fadd float %conv, %conv4
  %4 = load i32, i32* %i, align 4
  %5 = load i32, i32* %j, align 4
  %add5 = add nsw i32 %4, %5
  %add6 = add nsw i32 %add5, 1
  %conv7 = sitofp i32 %add6 to float
  %div = fdiv float %add, %conv7
  %6 = load i32, i32* %i, align 4
  %idxprom = sext i32 %6 to i64
  %arrayidx = getelementptr inbounds [6 x [8 x float]], [6 x [8 x float]]* %A, i64 0, i64 %idxprom
  %7 = load i32, i32* %j, align 4
  %idxprom8 = sext i32 %7 to i64
  %arrayidx9 = getelementptr inbounds [8 x float], [8 x float]* %arrayidx, i64 0, i64 %idxprom8
  store float %div, float* %arrayidx9, align 4
  %8 = load i32, i32* %i, align 4
  %idxprom10 = sext i32 %8 to i64
  %arrayidx11 = getelementptr inbounds [6 x [8 x float]], [6 x [8 x float]]* %B, i64 0, i64 %idxprom10
  %9 = load i32, i32* %j, align 4
  %idxprom12 = sext i32 %9 to i64
  %arrayidx13 = getelementptr inbounds [8 x float], [8 x float]* %arrayidx11, i64 0, i64 %idxprom12
  store float 0.000000e+00, float* %arrayidx13, align 4
  %10 = load i32, i32* %i, align 4
  %idxprom14 = sext i32 %10 to i64
  %arrayidx15 = getelementptr inbounds [6 x [8 x float]], [6 x [8 x float]]* %A, i64 0, i64 %idxprom14
  %11 = load i32, i32* %j, align 4
  %idxprom16 = sext i32 %11 to i64
  %arrayidx17 = getelementptr inbounds [8 x float], [8 x float]* %arrayidx15, i64 0, i64 %idxprom16
  %12 = load float, float* %arrayidx17, align 4
  %conv18 = fpext float %12 to double
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str, i64 0, i64 0), double %conv18)
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %13 = load i32, i32* %j, align 4
  %inc = add nsw i32 %13, 1
  store i32 %inc, i32* %j, align 4
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  %call19 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0))
  br label %for.inc20

for.inc20:                                        ; preds = %for.end
  %14 = load i32, i32* %i, align 4
  %inc21 = add nsw i32 %14, 1
  store i32 %inc21, i32* %i, align 4
  br label %for.cond

for.end22:                                        ; preds = %for.cond
  %call23 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0))
  %ptrToData = getelementptr inbounds %struct.TwoDMemrefF32, %struct.TwoDMemrefF32* %A_mem, i32 0, i32 0
  %arrayidx24 = getelementptr inbounds [6 x [8 x float]], [6 x [8 x float]]* %A, i64 0, i64 0
  %arrayidx25 = getelementptr inbounds [8 x float], [8 x float]* %arrayidx24, i64 0, i64 0
  store float* %arrayidx25, float** %ptrToData, align 8
  %alignedPtrToData = getelementptr inbounds %struct.TwoDMemrefF32, %struct.TwoDMemrefF32* %A_mem, i32 0, i32 1
  %arrayidx26 = getelementptr inbounds [6 x [8 x float]], [6 x [8 x float]]* %A, i64 0, i64 0
  %arrayidx27 = getelementptr inbounds [8 x float], [8 x float]* %arrayidx26, i64 0, i64 0
  store float* %arrayidx27, float** %alignedPtrToData, align 8
  %offset = getelementptr inbounds %struct.TwoDMemrefF32, %struct.TwoDMemrefF32* %A_mem, i32 0, i32 2
  store i64 0, i64* %offset, align 8
  %shape = getelementptr inbounds %struct.TwoDMemrefF32, %struct.TwoDMemrefF32* %A_mem, i32 0, i32 3
  %arrayinit.begin = getelementptr inbounds [2 x i64], [2 x i64]* %shape, i64 0, i64 0
  store i64 6, i64* %arrayinit.begin, align 8
  %arrayinit.element = getelementptr inbounds i64, i64* %arrayinit.begin, i64 1
  store i64 8, i64* %arrayinit.element, align 8
  %stride = getelementptr inbounds %struct.TwoDMemrefF32, %struct.TwoDMemrefF32* %A_mem, i32 0, i32 4
  %arrayinit.begin28 = getelementptr inbounds [2 x i64], [2 x i64]* %stride, i64 0, i64 0
  store i64 1, i64* %arrayinit.begin28, align 8
  %arrayinit.element29 = getelementptr inbounds i64, i64* %arrayinit.begin28, i64 1
  store i64 1, i64* %arrayinit.element29, align 8
  %ptrToData30 = getelementptr inbounds %struct.TwoDMemrefF32, %struct.TwoDMemrefF32* %B_mem, i32 0, i32 0
  %arrayidx31 = getelementptr inbounds [6 x [8 x float]], [6 x [8 x float]]* %B, i64 0, i64 0
  %arrayidx32 = getelementptr inbounds [8 x float], [8 x float]* %arrayidx31, i64 0, i64 0
  store float* %arrayidx32, float** %ptrToData30, align 8
  %alignedPtrToData33 = getelementptr inbounds %struct.TwoDMemrefF32, %struct.TwoDMemrefF32* %B_mem, i32 0, i32 1
  %arrayidx34 = getelementptr inbounds [6 x [8 x float]], [6 x [8 x float]]* %B, i64 0, i64 0
  %arrayidx35 = getelementptr inbounds [8 x float], [8 x float]* %arrayidx34, i64 0, i64 0
  store float* %arrayidx35, float** %alignedPtrToData33, align 8
  %offset36 = getelementptr inbounds %struct.TwoDMemrefF32, %struct.TwoDMemrefF32* %B_mem, i32 0, i32 2
  store i64 0, i64* %offset36, align 8
  %shape37 = getelementptr inbounds %struct.TwoDMemrefF32, %struct.TwoDMemrefF32* %B_mem, i32 0, i32 3
  %arrayinit.begin38 = getelementptr inbounds [2 x i64], [2 x i64]* %shape37, i64 0, i64 0
  store i64 6, i64* %arrayinit.begin38, align 8
  %arrayinit.element39 = getelementptr inbounds i64, i64* %arrayinit.begin38, i64 1
  store i64 8, i64* %arrayinit.element39, align 8
  %stride40 = getelementptr inbounds %struct.TwoDMemrefF32, %struct.TwoDMemrefF32* %B_mem, i32 0, i32 4
  %arrayinit.begin41 = getelementptr inbounds [2 x i64], [2 x i64]* %stride40, i64 0, i64 0
  store i64 1, i64* %arrayinit.begin41, align 8
  %arrayinit.element42 = getelementptr inbounds i64, i64* %arrayinit.begin41, i64 1
  store i64 1, i64* %arrayinit.element42, align 8
  call void @_mlir_ciface_load_store_2d(%struct.TwoDMemrefF32* %A_mem, %struct.TwoDMemrefF32* %B_mem)
  store i32 0, i32* %i, align 4
  br label %for.cond43

for.cond43:                                       ; preds = %for.inc61, %for.end22
  %15 = load i32, i32* %i, align 4
  %cmp44 = icmp slt i32 %15, 6
  br i1 %cmp44, label %for.body46, label %for.end63

for.body46:                                       ; preds = %for.cond43
  store i32 0, i32* %j, align 4
  br label %for.cond47

for.cond47:                                       ; preds = %for.inc57, %for.body46
  %16 = load i32, i32* %j, align 4
  %cmp48 = icmp slt i32 %16, 8
  br i1 %cmp48, label %for.body50, label %for.end59

for.body50:                                       ; preds = %for.cond47
  %17 = load i32, i32* %i, align 4
  %idxprom51 = sext i32 %17 to i64
  %arrayidx52 = getelementptr inbounds [6 x [8 x float]], [6 x [8 x float]]* %B, i64 0, i64 %idxprom51
  %18 = load i32, i32* %j, align 4
  %idxprom53 = sext i32 %18 to i64
  %arrayidx54 = getelementptr inbounds [8 x float], [8 x float]* %arrayidx52, i64 0, i64 %idxprom53
  %19 = load float, float* %arrayidx54, align 4
  %conv55 = fpext float %19 to double
  %call56 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str, i64 0, i64 0), double %conv55)
  br label %for.inc57

for.inc57:                                        ; preds = %for.body50
  %20 = load i32, i32* %j, align 4
  %inc58 = add nsw i32 %20, 1
  store i32 %inc58, i32* %j, align 4
  br label %for.cond47

for.end59:                                        ; preds = %for.cond47
  %call60 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0))
  br label %for.inc61

for.inc61:                                        ; preds = %for.end59
  %21 = load i32, i32* %i, align 4
  %inc62 = add nsw i32 %21, 1
  store i32 %inc62, i32* %i, align 4
  br label %for.cond43

for.end63:                                        ; preds = %for.cond43
  ret i32 0
}

declare dso_local i32 @printf(i8*, ...) #1

declare dso_local void @_mlir_ciface_load_store_2d(%struct.TwoDMemrefF32*, %struct.TwoDMemrefF32*) #1

attributes #0 = { noinline nounwind optnone uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.0 (git@github.com:llvm/llvm-project a8ef00af43ff7d1d2158d8715b8bab3bcb2d783a)"}
