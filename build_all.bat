@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d E:\dev\tools\two3

set ARCH=-arch=sm_75
set FLAGS=-O2

echo [1/7] Building test_two3...
nvcc %FLAGS% %ARCH% -o test_two3.exe test_two3.cu two3.cu
if %ERRORLEVEL% neq 0 (echo FAILED: test_two3 & exit /b 1)

echo [2/7] Building test_gain...
nvcc %FLAGS% %ARCH% -o test_gain.exe test_gain.cu two3.cu
if %ERRORLEVEL% neq 0 (echo FAILED: test_gain & exit /b 1)

echo [3/7] Building test_rope...
nvcc %FLAGS% %ARCH% -o test_rope.exe test_rope.cu two3.cu
if %ERRORLEVEL% neq 0 (echo FAILED: test_rope & exit /b 1)

echo [4/7] Building test_layer...
nvcc %FLAGS% %ARCH% -o test_layer.exe test_layer.cu two3.cu
if %ERRORLEVEL% neq 0 (echo FAILED: test_layer & exit /b 1)

echo [5/7] Building test_moe...
nvcc %FLAGS% %ARCH% -o test_moe.exe test_moe.cu two3.cu
if %ERRORLEVEL% neq 0 (echo FAILED: test_moe & exit /b 1)

echo [6/7] Building test_model...
nvcc %FLAGS% %ARCH% -o test_model.exe test_model.cu two3.cu
if %ERRORLEVEL% neq 0 (echo FAILED: test_model & exit /b 1)

echo [7/7] Building test_train...
nvcc %FLAGS% %ARCH% -o test_train.exe test_train.cu two3.cu
if %ERRORLEVEL% neq 0 (echo FAILED: test_train & exit /b 1)

echo.
echo BUILD OK — all 7 targets
