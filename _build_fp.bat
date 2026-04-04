@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d E:\dev\tools\two3
cl /O2 /Fe:ibc_precompute.exe ibc_precompute.c
if exist ibc_precompute.exe (echo BUILD OK) else (echo BUILD FAILED)
