#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <string>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <map>
#include <unistd.h>

#include "pathfinder_common.h"
#include "scene.h"
#include "solver.h"

using namespace std;

void gpuCheckError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

static int get_max_threads(int *pCount) {
	cudaDeviceProp prop;
	int count, maxThreads = 0;
	GPU_CHECKERROR(cudaGetDeviceCount(&count));
	int device = 0;
	for (int i = 0; i < count; i++) {
		GPU_CHECKERROR(cudaGetDeviceProperties(&prop, i));
		if (prop.maxThreadsPerBlock > maxThreads) {
			maxThreads = prop.maxThreadsPerBlock;
			device = i;
			*pCount = i;
		}
	}
	assert(maxThreads != 0);
	GPU_CHECKERROR(cudaSetDevice(device));
	return maxThreads;
}

static int get_sm_count(int device) {
	cudaDeviceProp prop;
	GPU_CHECKERROR(cudaGetDeviceProperties(&prop, device));
	return prop.multiProcessorCount;
}

static scene_t scene;
// Make a map to store human references
static map<string, human_t *> humanMap;
// Make a map to store goal references
static map<string, point_t *> goalMap;
// We need to serialize the scene
static point_t *linearGrid;

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "usage: %s <scene.map> <human.dsc>\n", *argv);
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
	// Get the maximum number of threads
	int numHumans = humanMap.size();
	int count;
	// Resource safety
	int threads = get_max_threads(&count) / 2;
	int sms = get_sm_count(count);
	int blocks = (int) ceil(((double) numHumans) / threads);
	debugPrintf("threads: %d, blocks: %d, sms: %d\n", threads, blocks, sms);
	// If we're asked to simulate more humans than we can, quit
	if (blocks > sms) {
		fprintf(stderr,
				"Unable to simulate more than %d humans on this configuration.\n",
				threads * 2 * blocks);
		freeScene(&scene, &humanMap);
		exit(2);
	}
	// We need more threads
	if (threads * blocks < numHumans) threads *= 2;

	// Make the grid linear
	linearizeGrid(&scene, &linearGrid);
	// Stick the humans in an array
	human_t *humans = (human_t *) malloc(numHumans * sizeof(human_t));
	int pos = 0;
	for (map<string, human_t *>::iterator it = humanMap.begin();
			it != humanMap.end(); ++it)
		humans[pos++] = *(it->second);

	// Copy over the grid, humans, paths, and statistics
	point_t *d_linearGrid;
	human_t *d_humans;
	stat_t *d_stats;
	nodeList_t **d_paths;
	// Allocate space to store the results
	void *results, *d_results;
	int *d_remainingHumans;
	unsigned int *remainingHumans;
	GPU_CHECKERROR(
			cudaMalloc(&d_linearGrid,
					sizeof(point_t) * scene.width * scene.height));
	GPU_CHECKERROR(cudaMalloc(&d_humans, sizeof(human_t) * numHumans));
	GPU_CHECKERROR(cudaMalloc(&d_stats, sizeof(stat_t) * numHumans));
	GPU_CHECKERROR(cudaMalloc(&d_paths, sizeof(nodeList_t *) * numHumans));
	GPU_CHECKERROR(
			cudaHostAlloc(&results,
					scene.width * scene.height * sizeof(simple_point_t)
							* numHumans + sizeof(int) * numHumans * 2,
					cudaHostAllocMapped));
	// Allocate space for checking if humans have all found a home
	GPU_CHECKERROR(
			cudaHostAlloc(&remainingHumans, sizeof(int), cudaHostAllocMapped));
	GPU_CHECKERROR(cudaHostGetDevicePointer(&d_results, results, 0));
	GPU_CHECKERROR(
			cudaHostGetDevicePointer(&d_remainingHumans, remainingHumans, 0));
	debugPrintf("pinned memory: %lu, width: %d, height: %d\n",
			scene.width * scene.height * sizeof(simple_point_t) * numHumans
					+ sizeof(int) * numHumans * 2, scene.width, scene.height);
	GPU_CHECKERROR(
			cudaMemcpy(d_linearGrid, linearGrid,
					sizeof(point_t) * scene.width * scene.height,
					cudaMemcpyHostToDevice));
	GPU_CHECKERROR(
			cudaMemcpy(d_humans, humans, sizeof(human_t) * numHumans,
					cudaMemcpyHostToDevice));
	GPU_CHECKERROR(
			cudaMemcpy(d_remainingHumans, &numHumans, sizeof(int),
					cudaMemcpyHostToDevice));
	GPU_CHECKERROR(cudaMemset(d_stats, 0, sizeof(stat_t) * numHumans));
	GPU_CHECKERROR(cudaMemset(d_paths, 0, sizeof(nodeList_t *) * numHumans));

	// Increase the heap size
	size_t limit;
	GPU_CHECKERROR(cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize));
	debugPrintf("max heap memory: %lu, current heap: %lu\n", limit, scene.width * scene.height * sizeof(node_t) * 8 * numHumans);
	GPU_CHECKERROR(
			cudaDeviceSetLimit(cudaLimitMallocHeapSize, (size_t) max(scene.width * scene.height * sizeof(node_t) * 8 * numHumans, limit)));
	// Launch the kernel
	int loops = (int) ceil(sqrt(scene.width * scene.width + scene.height * scene.height));
	while (*remainingHumans) {
		for (int i = 0; i < loops; i++) {
			solveScene<<<blocks, threads>>>(d_linearGrid, d_humans, d_stats, d_paths,
					scene.width, scene.height, numHumans, d_remainingHumans, d_results);
		}
	}
	GPU_CHECKERROR(cudaDeviceSynchronize());
	GPU_CHECKERROR(cudaGetLastError());

	// Free the memory
	GPU_CHECKERROR(cudaFree(d_linearGrid));
	GPU_CHECKERROR(cudaFree(d_humans));
	GPU_CHECKERROR(cudaFree(d_stats));
	GPU_CHECKERROR(cudaFree(d_paths));

	// Now, analyze the results
	analyzeResults(results, &humanMap, &scene, humans);
	GPU_CHECKERROR(cudaFreeHost(results));
	GPU_CHECKERROR(cudaFreeHost(remainingHumans));
	free(linearGrid);
	free(humans);
	freeScene(&scene, &humanMap);

	return 0;
}
