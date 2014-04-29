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

void analyzeResults(void *results, map<string, human_t *> *humans,
		scene_t *scene, unsigned int itrCnt, human_t *humanArr)
{
	int resultWidth = scene->width * scene->height * sizeof(simple_point_t)
			+ 2 * sizeof(int);
	printf("%u, %d\n", itrCnt, resultWidth);
	// For each result
	for (unsigned int i = 0; i < humans->size(); i++)
	{
		// Get the row
		void *rRow = (void *) (((char *) results) + resultWidth * i);
		// Determine the number of elements and collision count
		unsigned int *nums = (unsigned int *) (((char *) rRow)
				+ (resultWidth - sizeof(int) * 2));
		printf("human: %s, collisions: %u, elements: %u\n", humanArr[i].name,
				nums[0], nums[1]);
		// Print the path
		simple_point_t *point = ((simple_point_t *) rRow);
		printf("path: ");
		for (unsigned int k = 0; k < nums[1]; k++)
		{
			if (k == nums[1] - 1)
			{
				printf("(%d,%d)", point[k].x, point[k].y);
			}
			else
			{
				printf("(%d,%d), ", point[k].x, point[k].y);
			}
		}
		printf("\n");
	}
}

int linearizeGrid(scene_t *scene, point_t **linearGrid)
{
	// Allocate space for the linear grid
	*linearGrid = (point_t *) malloc(
			scene->width * scene->height * sizeof(point_t));
	for (int y = 0; y < scene->height; y++)
	{
		for (int x = 0; x < scene->width; x++)
		{
			// Copy over the point_ts
			(*linearGrid)[scene->width * y + x] = scene->grid[y][x];
			// Redirct the humans goals
			if ((*linearGrid)[scene->width * y + x].type == TEND)
			{
				debugPrintf("name: %s, this goal: %p, human goal: %p\n", (*linearGrid)[scene->width * y + x].hum->name, &(*linearGrid)[scene->width * y + x], (*linearGrid)[scene->width * y + x].hum->goal);
				(*linearGrid)[scene->width * y + x].hum->goalX = x;
				(*linearGrid)[scene->width * y + x].hum->goalY = y;
			}
		}
	}
	return 0;
}

void freeScene(scene_t *scene, map<string, human_t *> *humanMap)
{
	// Free the grid
	int rows = scene->height;
	for (int j = 0; j <= rows; j++)
		free(scene->grid[j]);
	free(scene->grid);
	// Free the humans
	for (map<string, human_t *>::iterator it = humanMap->begin();
			it != humanMap->end(); ++it)
	{
		free(it->second->name);
		free(it->second);
	}
}

int buildMap(FILE *mapFile, scene_t *scene, map<string, human_t *> *humanMap,
		map<string, point_t *> *goalMap)
{
	// Make a buffer and zero it
	char *buf = (char *) calloc(1, BUFSIZE * sizeof(char));
	// Make a 1x1 map for now
	point_t **grid = (point_t **) malloc(1 * sizeof(point_t *));
	grid[0] = (point_t *) calloc(1, 1 * sizeof(point_t));

	// Setup some parameters
	int max_x = 0;
	int max_y = 0;
	int curX = 0;

	// Read the file
	while (fgets(buf, BUFSIZE, mapFile) != NULL)
	{
		debugPrintf("%s", buf);
		// Get the current point
		char *tok = strtok(buf, mapDelims);
		do
		{
			// First, remove the newline if it exists
			char *nPos, *ePos;
			if ((nPos = strrchr(tok, '\n')))
				tok[nPos - tok] = 0;

			// If it ends with HUMCHAR, it is a human
			int tokLen = strlen(tok);
			char *sPos;
			if ((sPos = strrchr(tok, HUMCHAR)))
			{
				if ((int) (sPos - tok) == tokLen - 1)
				{
					grid[max_y][curX].type = THUM;
					// Make a human
					human_t *hum = (human_t *) calloc(1, sizeof(human_t));
					grid[max_y][curX].hum = hum;
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
					if (map_contains(goalMap, goal, cname))
					{
						hum->goal = goal->second;
						goal->second->hum = hum;
					}
					// Set its position
					hum->posX = curX;
					hum->posY = max_y;
					debugPrintf("name: %s, goal: %p, coord: %d,%d\n", hum->name, hum->goal, curX, max_y);
				}
			}
			// If it ends with ENDCHAR, it is a goal
			else if ((ePos = strrchr(tok, ENDCHAR)))
			{
				if ((int) (ePos - tok) == tokLen - 1)
				{
					grid[max_y][curX].type = TEND;
					// Allocate space for the name
					char *cname = (char *) malloc(tokLen * sizeof(char));
					// Copy the name
					strncpy(cname, tok, tokLen - 1);
					cname[tokLen - 1] = 0;
					// Add it to the map
					(*goalMap)[cname] = &(grid[max_y][curX]);
					// Attempt to find its human
					map<string, human_t *>::iterator hum;
					if (map_contains(humanMap, hum, cname))
					{
						grid[max_y][curX].hum = hum->second;
						hum->second->goal = &grid[max_y][curX];
					} debugPrintf("name: %s, human: %p, coord: %d,%d\n", cname, grid[max_y][curX].hum, curX, max_y);
					free(cname);
				}
			}
			// If it is equal to B or O, then it is an obstacle
			else if (strcmp(tok, OBSTR) == 0 || strcmp(tok, BLKSTR) == 0)
				grid[max_y][curX].type = TOBJ;
			// Otherwise, it is a path
			else
				grid[max_y][curX].type = TPATH;
			debugPrintf("%d,%d, type: %d\n", curX, max_y, grid[max_y][curX].type);

			// Increment current x, and max x if necessary
			if (++curX > max_x)
			{
				(max_x)++;
				// Make more space for the next element (on each row)
				for (int i = 0; i <= max_y; i++)
					grid[i] = (point_t *) realloc(grid[i],
							(curX + 1) * sizeof(point_t));
			}
			// If a new line was found, we are going to the next line
			if (nPos)
			{
				(max_y)++;
				curX = 0;
				// Increase the y depth
				grid = (point_t **) realloc(grid,
						(max_y + 1) * sizeof(point_t *));
				// Allocate a new row
				grid[max_y] = (point_t *) calloc(1, max_x * sizeof(point_t));
			}
		} while ((tok = strtok(NULL, mapDelims)));
	}
	free(buf);
	if (ferror(mapFile))
	{
		free(grid);
		return EREADERR;
	}
	scene->grid = grid;
	scene->height = max_y;
	scene->width = max_x;
	debugPrintf("max_x: %d, max_y: %d\n", max_x, max_y);
	return 0;
}
