call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d E:\dev\tools\two3
nvcc -O2 -arch=sm_75 -o test_train.exe test_train.cu two3.cu
if exist test_train.exe (echo BUILD OK) else (echo BUILD FAILED)
