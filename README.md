Base Paper used to improve upon:
><cite>Shrinivas Devshatwar, Madhur Amilkanthwar, and Rupesh Nasre. 2016. 
GPU centric extensions for parallel strongly connected components computation. In Proceedings of the 9th Annual Workshop on General Purpose Processing using Graphics Processing Unit (GPGPU '16). 
ACM, New York, NY, USA, 2-11. DOI: https://doi.org/10.1145/2884045.2884048</cite>

### Source code layout 

Source code can be found in src/ folder
and the relevant .h files in include/ folder.
Intermediate object files will be created 
in bin/ folder but the final binary named "scc" 
will be created in the current folder.

### How to compile?

1. The architecture to which the code should be compiled
can be set via GPU_ARCH variable. If nothing provided,
Makefile will compile the code for sm_35.

2. Path to NVCC compiler can be set via NVCC_PATH.
If nothing is provided, NVCC will be picked from
default location (i.e. location where CUDA toolkit installs
NVCC binary)

Final binary will be created in the current folder.

Examples:
    a. make 
       will compile SCC code for sm_35 architecture by
       picking binary from default location. 
    b. make NVCC_PATH=/path/to/nvcc GPU_ARCH=sm_37 
       will compile SCC code for sm_37 architecture by picking
       NVCC binary from /path/to/nvcc.


### How to execute? 

WARNING:
Please note that the input graph MUST be present in
GTgraph format. If not provided the results are undefined. 
Sample input graph file can be found in data/ folder.


./scc -a [g/h/x/y/d] -p [0/1] -q [0/1] -w [1/2/4/8/16/32] -f <file>
-a algorith to use   g - vHong, h - vSlota, x - wHong, y - wSlota
-p Trim-1 enable
-q Trim-2 enable
-w warp size(used only when warp based)
-f input file
