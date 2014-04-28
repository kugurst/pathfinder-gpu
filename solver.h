/*
 * solver.h
 *
 *  Created on: Apr 27, 2014
 *      Author: mark
 */

#ifndef SOLVER_H_
#define SOLVER_H_

#include "pathfinder_common.h"
#include "cuda.h"

typedef struct node_t {
	node_t *parent, *next;
	int x, y;
	int f, g, h;
} node_t;

typedef struct nodeList_t {
	node_t *node;
	nodeList_t *next;
	int size;
} nodeList_t;


typedef struct stat_t {
	nodeList_t *path;
	int collisions;
} stat_t;

__global__ void solveScene(point_t *, human_t *, stat_t *, int, int, int, int *, void *, unsigned int *);

#endif /* SOLVER_H_ */
