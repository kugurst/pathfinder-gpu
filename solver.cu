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
__device__ unsigned int count = 0;

__global__ void solveScene(point_t *grid, human_t *humans, stat_t *stats, int maxWidth, int maxHeight, int numHumans, int *remainingHumans, int threads)
{
	// We're done
	if (*remainingHumans == 0)
		return;
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= *remainingHumans && threadIdx.x != 0)
		return;
	debugPrintf("%d\n", gridDim.x);
	// Verify that the scene has been solved
	if (threadIdx.x == 0) {
		unsigned int value = atomicInc(&count, gridDim.x);
		isLastBlockDone = value == (gridDim.x - 1);
	}
}


