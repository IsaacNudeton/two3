@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d E:\dev\tools\two3
nvcc -O0 -arch=sm_75 -o test_model.exe test_model.cu two3.cu
if exist test_model.exe (echo BUILD OK) else (echo BUILD FAILED & exit /b 1)
E:\dev\tools\two3\test_model.exe
