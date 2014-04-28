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

__shared__ int remainingHumans;

typedef struct stat_t {

} stat_t;

typedef struct node {
	node *parent;
	int x, y;
	float f, g, h;
} node;

__global__ void solveScene(point_t *, human_t *, stat_t *, int, int, int, int *, int);

#endif /* SOLVER_H_ */
