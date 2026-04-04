@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d E:\dev\tools\two3
nvcc -O2 -o ibc_compile.exe ibc_compile.cu
if exist ibc_compile.exe (echo BUILD OK) else (echo BUILD FAILED)
