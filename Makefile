# Makefile — {2,3} Computing Architecture
# Linux build. For Windows see build.bat
#
# Target: RTX 2080 Super (SM 7.5, Turing)

NVCC     = nvcc
ARCH     = -arch=sm_75
FLAGS    = -O2 -Xcompiler -Wall
LDFLAGS  = -lcudart -lm

all: test_two3 test_gain test_rope test_layer test_moe test_model

test_two3: test_two3.cu two3.cu two3.h
	$(NVCC) $(ARCH) $(FLAGS) -o test_two3 test_two3.cu two3.cu $(LDFLAGS)

test_gain: test_gain.cu two3.cu two3.h gain.h
	$(NVCC) $(ARCH) $(FLAGS) -o test_gain test_gain.cu two3.cu $(LDFLAGS)

test_rope: test_rope.cu two3.cu two3.h rope.h
	$(NVCC) $(ARCH) $(FLAGS) -o test_rope test_rope.cu two3.cu $(LDFLAGS)

test_layer: test_layer.cu two3.cu two3.h gain.h rope.h activation.h
	$(NVCC) $(ARCH) $(FLAGS) -o test_layer test_layer.cu two3.cu $(LDFLAGS)

test_moe: test_moe.cu two3.cu two3.h moe.h
	$(NVCC) $(ARCH) $(FLAGS) -o test_moe test_moe.cu two3.cu $(LDFLAGS)

test_model: test_model.cu two3.cu two3.h gain.h rope.h activation.h moe.h model.h
	$(NVCC) $(ARCH) $(FLAGS) -o test_model test_model.cu two3.cu $(LDFLAGS)

run_all: all
	./test_two3 && ./test_gain && ./test_rope && ./test_layer && ./test_moe && ./test_model

clean:
	rm -f test_two3 test_gain test_rope test_layer test_moe test_model *.o

.PHONY: all run_all clean
