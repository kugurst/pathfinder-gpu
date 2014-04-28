/*
 * solver.cu
 *
 *  Created on: Apr 27, 2014
 *      Author: mark
 */

#include "cuda.h"

#include "solver.h"
#include "pathfinder_common.h"
#include "scene.h"

__shared__ bool isLastBlockDone;
__device__ unsigned int count1 = 0;
__device__ unsigned int count2 = 0;
__device__ unsigned int iterationCount = 0;

__device__ void syncAllThreads(unsigned int *syncCounter)
{
	unsigned int value = 0;
	__syncthreads();
	if (threadIdx.x == 0)
		value = atomicInc(syncCounter, gridDim.x - 1);
	volatile unsigned int* counter = syncCounter;
	do {} while (*counter > 0);
	if (value == gridDim.x - 1)
		iterationCount++;
}

__global__ void solveScene(point_t *grid, human_t *humans, stat_t *stats,
		int maxWidth, int maxHeight, int numHumans, int *remainingHumans)
{
	while (*remainingHumans != 0) {
		int id = threadIdx.x + blockDim.x * blockIdx.x;
		if (id >= *remainingHumans && threadIdx.x != 0)
			return;
		debugPrintf("%d\n", gridDim.x);
		// Verify that the scene has been solved
		syncAllThreads((iterationCount % 2 == 0) ? &count1 : &count2);
	}
}

