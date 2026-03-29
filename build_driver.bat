call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d E:\dev\tools\two3
nvcc -O0 -arch=sm_75 -o train_driver.exe train_driver.cu two3.cu
if exist train_driver.exe (echo BUILD OK) else (echo BUILD FAILED)
