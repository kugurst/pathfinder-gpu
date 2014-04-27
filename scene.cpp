/*
 * scene.cpp
 *
 *  Created on: Apr 26, 2014
 *      Author: mark
 */

#include <cstdlib>
#include <cstring>

#include "pathfinder_common.h"
#include "scene.h"

#define BUFSIZE 4096

int buildMap(FILE *mapFile, scene_t *scene, int *max_x, int *max_y)
{
	// Make a buffer and zero it
	char *buf = (char *) malloc(BUFSIZE * sizeof(char));
	memset(buf, 0, BUFSIZE * sizeof(char));
	// Make a 1x1 map for now
	point_t **grid = (point_t **) malloc(1 * sizeof(point_t *));
	grid[0] = (point_t *) malloc(1* sizeof(point_t));
	// Setup some parameters
	*max_x = *max_y = 1;
	point_t *curRow = grid[0];
	// Read the file
	while (fgets(buf, BUFSIZE, mapFile) != NULL) {
		printf("%s", buf);
	}
	if (ferror(mapFile))
		return EREADERR;

	return 0;
}
