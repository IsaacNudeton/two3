@echo off
REM build_driver.bat — Training Loop for {2,3}
REM Run from Developer Command Prompt for VS2022
REM
REM Usage:
REM   build_driver.bat              — Normal training build (SGD, -O3 optimized)
REM   build_driver.bat debug        — Debug build with reservoir/gain logging
REM   build_driver.bat debug-moon   — Debug + CPU Muon optimizer (very slow, for testing)
REM   build_driver.bat gpu-moon     — Release + GPU Muon optimizer (cuBLAS, fast)
REM   build_driver.bat verify-moon  — muon_ns_verify.exe (CPU vs GPU Newton-Schulz parity)
REM   build_driver.bat hadamard     — hadamard_ablation.exe (Hadamard pre-quant MSE ablation)
REM   build_driver.bat debug-exit   — debug + TWO3_DEBUG_EXIT_METRICS (per-layer cos + pseudo-byte log)
REM   build_driver.bat test-model-exit — test_model_exit.exe (TWO3_EARLY_EXIT, Test 7 parity)

echo.
echo  {2,3} Training Driver — Build
echo.

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d E:\dev\tools\two3

if "%1"=="debug" goto debug
if "%1"=="debug-moon" goto debug_moon
if "%1"=="gpu-moon" goto gpu_moon
if "%1"=="gpu-resident" goto gpu_resident
if "%1"=="gpu-sixq" goto gpu_sixq
if "%1"=="fp-embed" goto fp_embed
if "%1"=="binary" goto binary
if "%1"=="binary-workspace" goto binary_workspace
if "%1"=="binary-tc" goto binary_tc
if "%1"=="binary-tc-cb" goto binary_tc_cb
if "%1"=="lex-class" goto lex_class
if "%1"=="lex-full" goto lex_full
if "%1"=="infer-duo" goto infer_duo
if "%1"=="verify-moon" goto verify_moon
if "%1"=="hadamard" goto hadamard
if "%1"=="hadamard-ablation" goto hadamard
if "%1"=="debug-exit" goto debug_exit
if "%1"=="test-model-exit" goto test_model_exit
goto normal

:verify_moon
echo  Building muon_ns_verify.exe (Newton-Schulz CPU/GPU parity)
nvcc -O2 -arch=sm_75 -DTWO3_MUON_GPU -o muon_ns_verify.exe muon_ns_verify.cu two3.cu -lcublas
goto end_verify

:hadamard
echo  Building hadamard_ablation.exe (host reference, no two3.cu)
nvcc -O2 -o hadamard_ablation.exe hadamard_ablation.cu
goto end_hadamard

:binary
echo  Building BINARY (device-resident backward+optimizer + GPU attention, default fast path)
nvcc -O3 -arch=sm_75 -DTWO3_BINARY -DTWO3_BINARY_RESIDENT -DTWO3_ATTN_GPU -o train_driver.exe train_driver.cu two3.cu
goto end

:binary_workspace
echo  Building BINARY-WORKSPACE (legacy workspace-backed path, no device-resident optimizer)
nvcc -O3 -arch=sm_75 -DTWO3_BINARY -o train_driver.exe train_driver.cu two3.cu
goto end

:binary_tc
echo  Building BINARY-TC (resident + Tensor Core forward matmul)
nvcc -O3 -arch=sm_75 -DTWO3_BINARY -DTWO3_BINARY_RESIDENT -DTWO3_TENSOR_CORE -o train_driver.exe train_driver.cu two3.cu
goto end

:binary_tc_cb
echo  Building BINARY-TC-CB (full stack + ternary codebook embedding)
nvcc -O3 -arch=sm_75 -DTWO3_BINARY -DTWO3_BINARY_RESIDENT -DTWO3_TENSOR_CORE -DTWO3_TERNARY_CODEBOOK -o train_driver.exe train_driver.cu two3.cu
goto end

:lex_class
echo  Building LEX-CLASS (TC + identity + character class embedding)
nvcc -O3 -arch=sm_75 -DTWO3_BINARY -DTWO3_BINARY_RESIDENT -DTWO3_TENSOR_CORE -DTWO3_LEX_EMBED -DTWO3_LEX_CLASS_ONLY -o train_driver.exe train_driver.cu two3.cu
goto end

:lex_full
echo  Building LEX-FULL (TC + identity + class + depth + mode embedding)
nvcc -O3 -arch=sm_75 -DTWO3_BINARY -DTWO3_BINARY_RESIDENT -DTWO3_TENSOR_CORE -DTWO3_LEX_EMBED -o train_driver.exe train_driver.cu two3.cu
goto end

:infer_duo
echo  Building INFER-DUO (prompt-based inference tool)
nvcc -O3 -arch=sm_75 -DTWO3_BINARY -o infer_duo.exe infer_duo.cu two3.cu
goto end

:fp_embed
echo  Building FP-EMBED (fingerprint embedding + four ternary projections)
nvcc -O3 -arch=sm_75 -DTWO3_FP_EMBED -o train_driver.exe train_driver.cu two3.cu
goto end

:gpu_sixq
echo  Building GPU-SIXQ (GPU-resident + weighted loss + sparse optim + layer skip)
nvcc -O3 -arch=sm_75 -DTWO3_GPU_RESIDENT -DTWO3_SIX_Q -o train_driver.exe train_driver.cu two3.cu
goto end

:gpu_resident
echo  Building GPU-RESIDENT (latent on device, bulk H2D, -O3 optimized)
nvcc -O3 -arch=sm_75 -DTWO3_GPU_RESIDENT -o train_driver.exe train_driver.cu two3.cu
goto end

:gpu_moon
echo  Building GPU-MOON (Newton-Schulz via cuBLAS, -O3 optimized)
echo  Linking: cublas.lib
nvcc -O3 -arch=sm_75 -DTWO3_MUON_GPU -o train_driver.exe train_driver.cu two3.cu -lcublas
goto end

:debug_moon
echo  Building DEBUG (CPU Muon optimizer + logging, SLOW)
nvcc -O0 -arch=sm_75 -DTWO3_DEBUG_MOE -DTWO3_DEBUG_GAIN -DTWO3_USE_MUON_TERNARY -o train_driver.exe train_driver.cu two3.cu
goto end

:debug
echo  Building DEBUG (reservoir/gain logging)
nvcc -O0 -arch=sm_75 -DTWO3_DEBUG_MOE -DTWO3_DEBUG_GAIN -o train_driver.exe train_driver.cu two3.cu
goto end

:debug_exit
echo  Building DEBUG + exit probe (TWO3_DEBUG_EXIT_METRICS — verbose per forward)
nvcc -O0 -arch=sm_75 -DTWO3_DEBUG_MOE -DTWO3_DEBUG_GAIN -DTWO3_DEBUG_EXIT_METRICS -o train_driver.exe train_driver.cu two3.cu
goto end

:test_model_exit
echo  Building test_model_exit.exe (TWO3_EARLY_EXIT — Test 7 full vs early parity; no probe spam)
nvcc -O0 -arch=sm_75 -DTWO3_EARLY_EXIT -o test_model_exit.exe test_model.cu two3.cu
goto end_test_model_exit

:normal
echo  Building NORMAL (SGD optimizer, -O3 optimized)
nvcc -O3 -arch=sm_75 -o train_driver.exe train_driver.cu two3.cu
goto end

:end
if exist train_driver.exe (echo BUILD OK & set BUILDRC=0) else (echo BUILD FAILED & set BUILDRC=1)
goto print_help

:end_verify
if exist muon_ns_verify.exe (echo BUILD OK — run muon_ns_verify.exe & set BUILDRC=0) else (echo BUILD FAILED & set BUILDRC=1)
goto print_help

:end_hadamard
if exist hadamard_ablation.exe (echo BUILD OK — run hadamard_ablation.exe & set BUILDRC=0) else (echo BUILD FAILED & set BUILDRC=1)
goto print_help

:end_test_model_exit
if exist test_model_exit.exe (echo BUILD OK — run test_model_exit.exe & set BUILDRC=0) else (echo BUILD FAILED & set BUILDRC=1)
goto print_help

:print_help
echo.
if "%1"=="hadamard" (
    echo  Run: hadamard_ablation.exe
    echo   (MSE: ternary quant with vs without Hadamard; host reference)
) else if "%1"=="verify-moon" (
    echo  Run: muon_ns_verify.exe
    echo   (Compares CPU vs GPU Newton-Schulz; exit 0 if within tolerance)
) else if "%1"=="gpu-moon" (
    echo  Run: train_driver.exe data.txt --medium --seq-len 64 --batch 2 --log-every 10
    echo   (GPU Muon optimizer — Newton-Schulz on cuBLAS, ~2-5x SGD overhead)
) else if "%1"=="debug-moon" (
    echo  Run: train_driver.exe data.txt --medium --seq-len 64 --batch 2 --log-every 10
    echo   (WARNING: CPU Muon optimizer is ~100x slower than SGD)
) else if "%1"=="debug" (
    echo  Run: train_driver.exe data.txt --medium --seq-len 64 --batch 2 --log-every 10
    echo   (Debug logging enabled: expert counts, reservoir levels, depletion)
) else if "%1"=="debug-exit" (
    echo  Run: train_driver.exe data.txt --medium --seq-len 4 --batch 1 --log-every 5 --epochs 1
    echo   (Same as debug plus [exit_probe] lines each layer in train forward)
) else if "%1"=="test-model-exit" (
    echo  Run: test_model_exit.exe
    echo   (Test 7: argmax match rate and max L2 full vs early; edit nvcc line to add probe flags if needed)
) else (
    echo  Run: train_driver.exe data.txt --medium --seq-len 64 --batch 2 --log-every 10
    echo   (Normal training build, minimal logging)
)
echo.
exit /b %BUILDRC%
