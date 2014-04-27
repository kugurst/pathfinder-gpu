
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <string>
#include <map>
#include <unistd.h>

#include "pathfinder_common.h"
#include "scene.h"

using namespace std;

static scene_t scene;
// Make a map to store human references
static map<string, human_t *> humanMap;
// Make a map to store goal references
static map<string, point_t *> goalMap;
static int width, height;

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
	buildMap(mapFile, &scene, &width, &height, &humanMap, &goalMap);
	return 0;
}
