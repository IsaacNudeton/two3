@echo off
REM build.bat — {2,3} Computing Kernel
REM Run from Developer Command Prompt for VS2022
REM or wherever nvcc is in PATH
REM
REM Target: RTX 2080 Super (SM 7.5, Turing)

echo.
echo  {2,3} Computing Kernel — Build
echo  Two states. One substrate. No multiply.
echo.

nvcc -O2 -arch=sm_75 -o test_two3.exe test_two3.cu two3.cu
if %ERRORLEVEL% neq 0 (
    echo BUILD FAILED
    exit /b 1
)

echo BUILD OK — test_two3.exe
echo.
echo Run with: test_two3.exe
