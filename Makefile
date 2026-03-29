# Makefile — {2,3} Computing Kernel
# Linux build. For Windows see build.bat

NVCC     = nvcc
ARCH     = -arch=sm_75    # Turing (RTX 2080 Super)
FLAGS    = -O2 -Xcompiler -Wall
LDFLAGS  = -lcudart

all: test_two3

test_two3: test_two3.cu two3.cu two3.h
	$(NVCC) $(ARCH) $(FLAGS) -o test_two3 test_two3.cu two3.cu $(LDFLAGS)

run: test_two3
	./test_two3

clean:
	rm -f test_two3 *.o

.PHONY: all run clean
