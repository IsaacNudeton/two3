@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
cd /d E:\dev\tools\two3
nvcc -O3 -arch=sm_75 -DTWO3_BINARY -DTWO3_TENSOR_CORE -o test_fwd_parity_tc.exe test_binary_forward_parity.cu two3.cu 2>&1
if errorlevel 1 (echo BUILD FAILED fwd-tc & exit /b 1)
nvcc -O3 -arch=sm_75 -DTWO3_BINARY -DTWO3_TENSOR_CORE -o test_bwd_parity_tc.exe test_binary_backward_parity.cu two3.cu 2>&1
if errorlevel 1 (echo BUILD FAILED bwd-tc & exit /b 1)
nvcc -O3 -arch=sm_75 -DTWO3_BINARY -DTWO3_TENSOR_CORE -o test_tc_parity.exe test_two3_tensor_core.cu two3.cu 2>&1
if errorlevel 1 (echo BUILD FAILED tc & exit /b 1)
echo.
echo === FORWARD PARITY (Tensor Core path) ===
E:\dev\tools\two3\test_fwd_parity_tc.exe
echo.
echo === BACKWARD PARITY (Tensor Core path) ===
E:\dev\tools\two3\test_bwd_parity_tc.exe
echo.
echo === WMMA vs BITMASK DIRECT PARITY ===
E:\dev\tools\two3\test_tc_parity.exe
exit /b 0
