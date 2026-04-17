# Prefer the local CUDA toolkit, and pin host compiler to GCC 12 by default
# (CUDA 13.1 may reject GCC > 12 unless you override).
CUDA_HOME ?= /usr/local/cuda
HOST_CXX ?= g++-12
NVCC = $(CUDA_HOME)/bin/nvcc -ccbin $(HOST_CXX) -lcudadevrt -rdc=true -O3 #--ptxas-options=-v
# NVCC = nvcc   -lcudadevrt -rdc=true -O3 -g -G #--ptxas-options=-v
CFLAGS = -c -Wall -O6
# CFLAGS = -c -Wall -O6 -g
EXEFLAG = -O6
OPENMP = -Xcompiler -fopenmp

library = -lcudadevrt

bisson: tc.cu preTC.cu 
	$(NVCC) -o bisson tc.cu $(OPENMP)
.PHONY : clean
clean:
	rm bisson
