
ifndef GPU_ARCH
GPU_ARCH=sm_35
endif

ifndef NVCC_PATH
NVCC_PATH=nvcc
endif

CFLAGS = -arch=$(GPU_ARCH)
INCLUDE = -I./include

all : scc
	$(info Code compiled successfully!)

scc: scc.o scc_kernels.o load.o scc_Coloring.o scc_WCC.o
	$(NVCC_PATH) $(INCLUDE) $(CFLAGS) -o scc bin/scc.o bin/scc_kernels.o bin/load.o bin/scc_Coloring.o bin/scc_WCC.o

scc.o:
	$(NVCC_PATH) $(INCLUDE) $(CFLAGS) -c src/scc.cu -o bin/scc.o

scc_Coloring.o:  
	$(NVCC_PATH) $(INCLUDE) $(CFLAGS) -c src/scc_Coloring.cu -o bin/scc_Coloring.o

scc_WCC.o: 
	$(NVCC_PATH) $(INCLUDE) $(CFLAGS) -c src/scc_WCC.cu -o bin/scc_WCC.o

load.o: 
	$(NVCC_PATH) $(INCLUDE) $(CFLAGS) -c src/load.cpp -o bin/load.o
	
scc_kernels.o: 
	$(NVCC_PATH) $(INCLUDE) $(CFLAGS) -c src/scc_kernels.cu -o bin/scc_kernels.o

clean:
	rm -rf bin/*.o scc
