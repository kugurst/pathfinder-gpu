
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <string>
#include <cmath>
#include <cassert>
#include <map>
#include <unistd.h>

#include "pathfinder_common.h"
#include "scene.h"
#include "solver.h"

using namespace std;

void gpuCheckError( cudaError_t err,
                         const char *file,
                         int line ) {
   if (err != cudaSuccess) {
       printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
              file, line );
       exit( EXIT_FAILURE );
   }
}

static int get_max_threads()
{
    cudaDeviceProp prop;
    int count, maxThreads = 0;
    GPU_CHECKERROR(cudaGetDeviceCount(&count));
    int device = 0;
    for (int i = 0; i < count; i++) {
        GPU_CHECKERROR(cudaGetDeviceProperties(&prop, i));
        if (prop.maxThreadsPerBlock > maxThreads) {
            maxThreads = prop.maxThreadsPerBlock;
            device = i;
        }
    }
    assert(maxThreads != 0);
    GPU_CHECKERROR(cudaSetDevice(device));
    return maxThreads;
}

static scene_t scene;
// Make a map to store human references
static map<string, human_t *> humanMap;
// Make a map to store goal references
static map<string, point_t *> goalMap;
// We need to serialize the scene
static point_t *linearGrid;

int main(int argc, char** argv)
{
	if (argc != 3) {
		fprintf(stderr, "usage: %s <scene.map> <human.dsc>", *argv);
		exit(1);
	}
	FILE *mapFile = fopen(argv[1], "r");
	if (!mapFile) {
		perror(argv[1]);
		return ENOFILE;
	}
	buildMap(mapFile, &scene, &humanMap, &goalMap);
	// We're done with the file, so close it
	fclose(mapFile);

	// Make the grid linear
	linearizeGrid(&scene, &linearGrid);
	// Stick the humans in an array
	int numHumans = humanMap.size();
	human_t *humans = (human_t *) malloc(numHumans * sizeof(human_t));
	int pos = 0;
	for (map<string, human_t *>::iterator it = humanMap.begin(); it != humanMap.end(); ++it)
		humans[pos++] = *(it->second);

	// Copy over the grid, humans, and statistics
	point_t *d_linearGrid;
	human_t *d_humans;
	stat_t *d_stats;
	// Allocate space to store the results
	void *results, *d_results;
	int *d_remainingHumans;
	unsigned int *d_itrCnt, itrCnt, zero = 0;
	GPU_CHECKERROR(cudaMalloc(&d_linearGrid, sizeof(point_t) * scene.width * scene.height));
	GPU_CHECKERROR(cudaMalloc(&d_humans, sizeof(human_t) * numHumans));
	GPU_CHECKERROR(cudaMalloc(&d_stats, sizeof(stat_t) * numHumans));
	GPU_CHECKERROR(cudaMalloc(&d_remainingHumans, sizeof(int)));
	GPU_CHECKERROR(cudaMalloc(&d_itrCnt, sizeof(int)));
	GPU_CHECKERROR(cudaHostAlloc(&results, scene.width * scene.height * sizeof(simple_point_t) * numHumans + sizeof(int) * numHumans * 2, cudaHostAllocMapped));
	GPU_CHECKERROR(cudaHostGetDevicePointer(&d_results, results, 0));
	debugPrintf("pinned memory: %lu, width: %d, height: %d\n", scene.width * scene.height * sizeof(simple_point_t) * numHumans + sizeof(int) * numHumans * 2, scene.width, scene.height);
	GPU_CHECKERROR(cudaMemcpy(d_linearGrid, linearGrid, sizeof(point_t) * scene.width * scene.height, cudaMemcpyHostToDevice));
	GPU_CHECKERROR(cudaMemcpy(d_humans, humans, sizeof(human_t) * numHumans, cudaMemcpyHostToDevice));
	GPU_CHECKERROR(cudaMemcpy(d_remainingHumans, &numHumans, sizeof(int), cudaMemcpyHostToDevice));
	GPU_CHECKERROR(cudaMemcpy(d_itrCnt, &zero, sizeof(int), cudaMemcpyHostToDevice));

	// Get the maximum number of threads
	int threads = get_max_threads();
	int blocks = (int) ceil(((double) numHumans) / threads);
	// Increase the heap size
	GPU_CHECKERROR(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 200 * 1024 * 1024));
	// Launch the kernel
	solveScene<<<blocks, threads>>>(d_linearGrid, d_humans, d_stats, scene.width, scene.height, numHumans, d_remainingHumans, d_results, d_itrCnt);
	GPU_CHECKERROR(cudaMemcpy(&itrCnt, d_itrCnt, sizeof(int), cudaMemcpyDeviceToHost));
	GPU_CHECKERROR(cudaDeviceSynchronize());

	// Free the memory
	GPU_CHECKERROR(cudaFree(d_linearGrid));
	GPU_CHECKERROR(cudaFree(d_humans));
	GPU_CHECKERROR(cudaFree(d_stats));
	GPU_CHECKERROR(cudaFree(d_remainingHumans));

	// Now, analyze the results
	analyzeResults(results, &humanMap, &scene, itrCnt, humans);
	GPU_CHECKERROR(cudaFreeHost(results));
	free(linearGrid);
	free(humans);
	freeScene(&scene, &humanMap);

	return 0;
}
