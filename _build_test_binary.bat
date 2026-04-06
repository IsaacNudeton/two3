@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d E:\dev\tools\two3
cl /O2 /Fe:test_binary.exe test_binary.c
if exist test_binary.exe (E:\dev\tools\two3\test_binary.exe) else (echo BUILD FAILED)
