@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d E:\dev\tools\two3
nvcc -O0 -g -arch=sm_75 -DTWO3_BINARY -o train_binary_debug.exe train_driver.cu two3.cu
if not exist train_binary_debug.exe (echo BUILD FAILED & exit /b 1)
echo BUILD OK
E:\dev\tools\two3\train_binary_debug.exe shakespeare.txt --medium --seq-len 4 --batch 1 --log-every 1 --epochs 1
