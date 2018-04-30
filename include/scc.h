#ifndef SCC_H
#define SCC_H

#include<iostream>
#include<fstream>
#include<stdint.h>

//#define _DEBUG
#define BLOCKSIZE 512
#define MaxXDimOfGrid 65535

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define BIT_SHIFT ((unsigned)1 << 31)

__host__ __device__ inline bool isForwardVisited(uint8_t tags) { return ( tags & 1); }
__host__ __device__ inline bool isForwardPropagate(uint8_t tags ) { return (tags & 4); }
__host__ __device__ inline bool isBackwardVisited(uint8_t tags) { return (tags & 2); }
__host__ __device__ inline bool isBackwardPropagate(uint8_t tags) { return ( tags & 8); }
__host__ __device__ inline void setForwardVisitedBit(uint8_t *tags) { *tags = ( *tags | 1); };
__host__ __device__ inline void setForwardPropagateBit(uint8_t *tags) { *tags = ( *tags | 4); };
__host__ __device__ inline void setBackwardVisitedBit(uint8_t *tags) { *tags = ( *tags | 2); };
__host__ __device__ inline void setBackwardPropagateBit(uint8_t *tags) { *tags = ( *tags | 8); };
__host__ __device__ inline void clearForwardVisitedBit(uint8_t *tags) { *tags = (*tags & ~1); };
__host__ __device__ inline void clearForwardPropagateBit(uint8_t *tags) { *tags = (*tags & ~4); };
__host__ __device__ inline void clearBackwardVisitedBit(uint8_t *tags) { *tags = (*tags & ~2); };
__host__ __device__ inline void clearBackwardPropagateBit(uint8_t *tags) { *tags = (*tags & ~8); };
__host__ __device__ inline bool isRangeSet(uint8_t tags) { return ( tags & 16); }
__host__ __device__ inline void rangeSet(uint8_t *tags) { *tags = ( *tags | 16); };
__host__ __device__ inline void setTrim1(uint8_t *tags) { *tags = ( *tags | 32); };
__host__ __device__ inline bool isTrim1(uint8_t tags) { return ( tags & 32); }
__host__ __device__ inline void setTrim2(uint8_t *tags) { *tags = ( *tags | 64); };
__host__ __device__ inline bool isTrim2(uint8_t tags) { return ( tags & 64); }
__host__ __device__ inline void setPivot(uint8_t *tags) { *tags = ( *tags | 128); };
__host__ __device__ inline bool isPivot(uint8_t tags) { return ( tags & 128); }
__host__ __device__ inline bool hasIncomingEdge(uint8_t flags) { return ( flags & 1); }
__host__ __device__ inline bool hasOutgoingEdge(uint8_t flags) { return (flags & 2); }
__host__ __device__ inline void setIncomingEdge(uint8_t *flags) { *flags = ( *flags | 1); };
__host__ __device__ inline void setOutgoingEdge(uint8_t *flags) { *flags = ( *flags | 2); };

void wSlota(uint32_t CSize, uint32_t RSize, uint32_t *Fc, uint32_t *Fr, uint32_t * Bc, uint32_t * Br, bool t1, bool t2, int warpSize);
void vSlota(uint32_t CSize, uint32_t RSize, uint32_t *Fc, uint32_t *Fr, uint32_t * Bc, uint32_t * Br, bool t1, bool t2);
void wHong(uint32_t CSize, uint32_t RSize, uint32_t *Fc, uint32_t *Fr, uint32_t * Bc, uint32_t * Br, bool t1, bool t2, int warpSize);
void vHong(uint32_t CSize, uint32_t RSize, uint32_t *Fc, uint32_t *Fr, uint32_t * Bc, uint32_t * Br, bool t1, bool t2);
void detectSCC(uint32_t CSize, uint32_t RSize, uint32_t *Fc, uint32_t *Fr, uint32_t * Bc, uint32_t * Br, uint32_t * Pr, bool t1, bool t2);
#endif
