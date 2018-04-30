#ifndef SCC_KERNELS_H
#define SCC_KERNELS_H

#include "scc.h"
__global__ void selectPivotsLocalNew(const uint32_t *range, uint8_t *tags, const uint32_t num_rows, uint32_t** pivot_field, const int max_pivot_count, uint32_t *Pr, bool *auxRange);
__global__ void pollForPivotsLocalNew(const uint32_t *range, const uint8_t *tags, const uint32_t num_rows, uint32_t** pivot_field, const int max_pivot_count, const uint32_t *Fr, const uint32_t *Br, uint32_t *Pr, bool volatile *terminate, bool *auxRange);
__global__ void selectPivots(const uint32_t *range, uint8_t *tags, const uint32_t num_rows, const uint32_t *pivot_field, const int max_pivot_count);
__global__ void pollForPivots(const uint32_t *range, const uint8_t *tags, const uint32_t num_rows, uint32_t* pivot_field, const int max_pivot_count, const uint32_t *Fr, const uint32_t *Br);
__global__ void update(uint32_t *range, uint8_t *tags, const uint32_t num_rows, bool volatile *terminate);
__global__ void trim1(const uint32_t *range, uint8_t *tags, const uint32_t *Fc, const uint32_t *Fr, const uint32_t *Bc, const uint32_t *Br, const uint32_t num_rows, bool volatile *terminate);
__global__ void trim2(const uint32_t *range, uint8_t *tags, const uint32_t *Fc, const uint32_t *Fr, const uint32_t *Bc, const uint32_t *Br, const uint32_t num_rows);
__global__ void fwd(const uint32_t *Fc, const uint32_t *Fr, const uint32_t *range, uint8_t *tags, const uint32_t num_rows, bool volatile *terminate);
__global__ void fwdLocal(const uint32_t *Fc, const uint32_t *Fr, const uint32_t *range, uint8_t *tags, uint32_t *Pr, const uint32_t num_rows, bool volatile *terminate, bool *auxRange);
__global__ void bwd(const uint32_t *Bc, const uint32_t *Br, const uint32_t *range, uint8_t *tags, const uint32_t num_rows, bool volatile *terminate);
__global__ void bwdLocal(const uint32_t *Bc, const uint32_t *Br, const uint32_t *range, uint8_t *tags, uint32_t *Pr, const uint32_t num_rows, bool volatile *terminate, bool *auxRange);
__global__ void pollForPivotsLocal(const uint32_t *range, const uint8_t *tags, const uint32_t num_rows, uint32_t** pivot_field, const int max_pivot_count, const uint32_t *Fr, const uint32_t *Br, uint32_t *Pr, bool volatile *terminate, bool *auxRange);
__global__ void selectPivotsLocal(const uint32_t *range, uint8_t *tags, const uint32_t num_rows, uint32_t** pivot_field, const int max_pivot_count, uint32_t *Pr, bool *auxRange);
__global__ void computeInDegree(const uint8_t *tags, const uint32_t num_rows, uint32_t* Pr, const uint32_t *Fr, const uint32_t *Br, bool *Occ, bool volatile *terminate);
__global__ void assignUniqueRange(uint32_t *range, const uint8_t *tags, const uint32_t num_rows);
__global__ void propagateRange1(const uint32_t *Fc, const uint32_t *Fr, uint32_t *range, const uint8_t *tags, const uint32_t num_rows, bool volatile *terminate);
__global__ void propagateRange2(uint32_t *range, const uint8_t *tags, const uint32_t num_rows, bool volatile *terminate);
__global__ void assignPartitionRange(uint32_t *range, const uint8_t *tags, uint32_t *Pr, const uint32_t num_rows);
__global__ void pollForFirstPivot(const uint8_t *tags, const uint32_t num_rows, uint32_t* pivot_field, const uint32_t *Fr, const uint32_t *Br);__global__ void selectFirstPivot(uint8_t *tags, const uint32_t num_rows, const uint32_t *pivot_field);
__global__ void computeOutDegree(const uint8_t *tags, const uint32_t num_rows, uint32_t* Pr, const uint32_t *Fr, const uint32_t *Fc, bool *Occ, bool volatile *terminate);
__global__ void getMaxRange(uint32_t *range, uint32_t *Pr, uint32_t *Rm, const uint32_t num_rows, uint8_t *tags, bool volatile *terminate);
__global__ void shiftRange(uint32_t *range, uint32_t *Pr, uint32_t *Rm, const uint32_t num_rows, uint8_t *tags);
__global__ void fwdProp(const uint32_t *Fc, const uint32_t *Fr, uint32_t *range, const uint8_t *tags, const uint32_t num_rows, bool volatile *terminate, uint32_t *Pr, bool *Occ);
__global__ void updatePr(uint32_t *Pr, const uint32_t num_rows, bool volatile *terminate, uint8_t *tags);
__global__ void updateLocal(uint32_t *range, uint8_t *tags, const uint32_t num_rows, bool volatile *terminate, bool *auxRange);
__global__ void identifyTransEdges(const uint32_t *Fc, const uint32_t *Fr, uint32_t *range, const uint8_t *tags, const uint32_t num_rows, uint32_t *Pr, bool *Occ);
__global__ void fwdRc(const uint32_t *Fc, const uint32_t *Fr, const uint32_t *range, uint8_t *tags, uint32_t *Pr, const uint32_t num_rows, bool volatile *terminate);
__global__ void bwdRc(const uint32_t *Bc, const uint32_t *Br, const uint32_t *range, uint8_t *tags, uint32_t *Pr, const uint32_t num_rows, bool volatile *terminate);
__global__ void resetTag(uint32_t *range, uint8_t *tags, const uint32_t num_rows, const int i);
//coloring
__global__ void updateColoring(uint8_t *tags, const uint32_t num_rows, bool volatile *terminate);
__global__ void fwdColoring(const uint32_t *Fc, const uint32_t *Fr, const uint32_t *range, uint8_t *tags, const uint32_t num_rows, bool volatile *terminate);
__global__ void selectPivotColoring(const uint32_t *range, uint8_t *tags, const uint32_t num_rows);
__global__ void colorPropagation(const uint32_t *Fc, const uint32_t *Fr, uint32_t *range, const uint8_t *tags, const uint32_t num_rows, bool volatile *terminate);

template<int w>
int fun(int x){
        int k = w + x;
        return k;
    }

template<int WARPSIZE>
__global__ void fwd_warp(const uint32_t *Fc, const uint32_t *Fr, const uint32_t *range, uint8_t *tags, const uint32_t num_rows, bool volatile *terminate){

    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t wRow = (row/WARPSIZE) + 1;
    uint8_t myTag;

    if (wRow > num_rows || isRangeSet(myTag = tags[wRow]) || isForwardPropagate(myTag) || !isForwardVisited(myTag))
        return;

    uint32_t myRange = range[wRow];
    uint32_t cnt = Fr[wRow + 1] - Fr[wRow];
    const uint32_t *nbrs = &Fc[Fr[wRow]];

    bool end = true;
    for ( uint32_t i = (row % WARPSIZE); i < cnt; i+=WARPSIZE ) {
        uint32_t index = nbrs[i];
        uint8_t nbrTag = tags[index];

        if(isRangeSet(nbrTag) || isForwardVisited(nbrTag) || range[index] != myRange)
            continue;

        setForwardVisitedBit(&tags[index]);
        end = false;
    }
    setForwardPropagateBit(&tags[wRow]);
    if (!end)
        *terminate = false;
}


template<int WARPSIZE>
__global__ void bwd_warp(const uint32_t *Bc, const uint32_t *Br, const uint32_t *range, uint8_t *tags, const uint32_t num_rows, bool volatile *terminate){

    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t wRow = (row/WARPSIZE) + 1;
    uint8_t myTag;

    if (wRow > num_rows || isRangeSet(myTag = tags[wRow]) || isBackwardPropagate(myTag) || !isBackwardVisited(myTag))
        return;

    uint32_t myRange = range[wRow];
    uint32_t cnt = Br[wRow + 1] - Br[wRow];
    const uint32_t *nbrs = &Bc[Br[wRow]];

    bool end = true;
    for ( uint32_t i = (row % WARPSIZE); i < cnt; i+=WARPSIZE ) {
        uint32_t index = nbrs[i];
        uint8_t nbrTag = tags[index];

        if(isRangeSet(nbrTag) || isBackwardVisited(nbrTag) || range[index] != myRange )
            continue;

        setBackwardVisitedBit(&tags[index]);
        end = false;
    }
    setBackwardPropagateBit(&tags[wRow]);
    if (!end)
        *terminate = false;
}


#endif
