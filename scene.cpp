/*
 * scene.cpp
 *
 *  Created on: Apr 26, 2014
 *      Author: mark
 */

#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <map>

#include "pathfinder_common.h"
#include "scene.h"

#define BUFSIZE 4096
#define map_contains(map, itr, name) 	(itr = map->find(name)) != map->end()

using namespace std;

static const char *mapDelims = " \t";

int buildMap(FILE *mapFile, scene_t *scene, int *max_x, int *max_y, map<string, human_t *> *humanMap, map<string, point_t *> *goalMap)
{
	// Make a buffer and zero it
	char *buf = (char *) calloc(1, BUFSIZE * sizeof(char));
	// Make a 1x1 map for now
	point_t **grid = (point_t **) malloc(1 * sizeof(point_t *));
	grid[0] = (point_t *) calloc(1, 1* sizeof(point_t));


	// Setup some parameters
	*max_x = 0;
	*max_y = 0;
	int curX = 0;

	// Read the file
	while (fgets(buf, BUFSIZE, mapFile) != NULL) {
		printf("%s", buf);
		// Get the current point
		char *tok = strtok(buf, mapDelims);
		do {
			// First, remove the newline if it exists
			char *nPos;
			if ((nPos = strrchr(tok, '\n')))
				tok[nPos - tok] = 0;

			// If it ends with HUMCHAR, it is a human
			int tokLen = strlen(tok);
			char *sPos;
			if ((sPos = strrchr(tok, HUMCHAR)))
				if ((int) (sPos - tok) == tokLen - 1) {
					grid[*max_y][curX].type = THUM;
					// Make a human
					human_t *hum = (human_t *) calloc(1, sizeof(human_t));
					grid[*max_y][curX].hum = hum;
					// Allocate space for the name
					char *cname = (char *) malloc(tokLen * sizeof(char));
					// Copy the name
					strncpy(cname, tok, tokLen - 1);
					cname[tokLen - 1] = 0;
					hum->name = cname;
					// Add it to the map
					(*humanMap)[cname] = hum;
					// Attempt to find its goal
					map<string, point_t *>::iterator goal;
					if (map_contains(goalMap, goal, cname)) {
						hum->goal = goal->second;
						goal->second->hum = hum;
					}
					printf("name: %s, goal: %p, coord: %d,%d\n", hum->name, hum->goal, curX, *max_y);
				}

			// If it ends with ENDCHAR, it is a goal
			char *ePos;
			if ((ePos = strrchr(tok, ENDCHAR)))
				if ((int) (ePos - tok) == tokLen - 1) {
					grid[*max_y][curX].type = TEND;
					// Allocate space for the name
					char *cname = (char *) malloc(tokLen * sizeof(char));
					// Copy the name
					strncpy(cname, tok, tokLen - 1);
					cname[tokLen - 1] = 0;
					// Add it to the map
					(*goalMap)[cname] = &(grid[*max_y][curX]);
					// Attempt to find its human
					map<string, human_t *>::iterator hum;
					if(map_contains(humanMap, hum, cname)) {
						grid[*max_y][curX].hum = hum->second;
						hum->second->goal = &grid[*max_y][curX];
					}
					printf("name: %s, human: %p, coord: %d,%d\n", cname, grid[*max_y][curX].hum, curX, *max_y);
					free(cname);
				}

			// If it is equal to B or O, then it is an obstacle
			if (strcmp(tok, OBCHAR) == 0 || strcmp(tok, BLKCHAR) == 0) {
				grid[*max_y][curX].type = TOBJ;
			}
			printf("%d,%d\n", curX, *max_y);

			// Increment current x, and max x if necessary
			if (++curX > *max_x) {
				(*max_x)++;
				// Make more space for the next element (on each row)
				for (int i = 0; i <= *max_y; i++)
					grid[i] = (point_t *) realloc(grid[i], curX * sizeof(point_t));
			}
			// If a new line was found, we are going to the next line
			if (nPos) {
				(*max_y)++;
				curX = 0;
				// Increase the y depth
				grid = (point_t **) realloc(grid, (*max_y + 1) * sizeof(point_t *));
				// Allocate a new row
				grid[*max_y] = (point_t *) calloc(1, *max_x * sizeof(point_t));
			}
		} while ((tok = strtok(NULL, mapDelims)));
	}
	free(buf);
	if (ferror(mapFile)) {
		free(grid);
		return EREADERR;
	}
	scene->grid = grid;
	printf("max_x: %d, max_y: %d\n", *max_x, *max_y);
	return 0;
}
