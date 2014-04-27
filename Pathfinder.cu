
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <unistd.h>

#include "pathfinder_common.h"
#include "scene.h"

static scene_t scene;
static int width, height;

int main(int argc, char** argv)
{
	if (argc != 2) {
		fprintf(stderr, "usage: %s <scene.map>", *argv);
		exit(1);
	}
	FILE *mapFile = fopen(argv[1], "r");
	if (!mapFile) {
		perror(argv[1]);
		return ENOFILE;
	}
	buildMap(mapFile, &scene, &width, &height);
	return 0;
}
