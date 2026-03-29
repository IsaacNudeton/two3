@echo off
REM build.bat — {2,3} Computing Architecture
REM Run from Developer Command Prompt for VS2022
REM Target: RTX 2080 Super (SM 7.5, Turing)

echo.
echo  {2,3} Computing Architecture — Build
echo  Two states. One substrate. No multiply.
echo.

set ARCH=-arch=sm_75
set FLAGS=-O2

echo [1/6] Building test_two3...
nvcc %FLAGS% %ARCH% -o test_two3.exe test_two3.cu two3.cu
if %ERRORLEVEL% neq 0 (echo FAILED: test_two3 & exit /b 1)

echo [2/6] Building test_gain...
nvcc %FLAGS% %ARCH% -o test_gain.exe test_gain.cu two3.cu
if %ERRORLEVEL% neq 0 (echo FAILED: test_gain & exit /b 1)

echo [3/6] Building test_rope...
nvcc %FLAGS% %ARCH% -o test_rope.exe test_rope.cu two3.cu
if %ERRORLEVEL% neq 0 (echo FAILED: test_rope & exit /b 1)

echo [4/6] Building test_layer...
nvcc %FLAGS% %ARCH% -o test_layer.exe test_layer.cu two3.cu
if %ERRORLEVEL% neq 0 (echo FAILED: test_layer & exit /b 1)

echo [5/6] Building test_moe...
nvcc %FLAGS% %ARCH% -o test_moe.exe test_moe.cu two3.cu
if %ERRORLEVEL% neq 0 (echo FAILED: test_moe & exit /b 1)

echo [6/6] Building test_model...
nvcc %FLAGS% %ARCH% -o test_model.exe test_model.cu two3.cu
if %ERRORLEVEL% neq 0 (echo FAILED: test_model & exit /b 1)

echo.
echo BUILD OK — all 6 targets
echo.
echo Run: test_model.exe (Layer 2 verification)
