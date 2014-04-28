/*
 * pathfinder_common.h
 *
 *  Created on: Apr 26, 2014
 *      Author: mark
 */
#ifndef PATHFINDER_COMMON_H_
#define PATHFINDER_COMMON_H_

#include "cuda_runtime.h"

#include <cstdio>

#define ENOFILE -2
#define EREADERR -3
#define EBADFORM -4

#define TPATH 1
#define TOBJ 2
#define THUM 3
#define TEND 4

#define HUMCHAR 'S'
#define ENDCHAR 'E'
#define OBSTR "O"
#define BLKSTR "B"

//#define DEBUG_PATH
#ifdef DEBUG_PATH
	#define debugPrintf(...)	printf(__VA_ARGS__);
#else
	#define debugPrintf(...)
#endif
// handy error macro:
#define GPU_CHECKERROR( err ) (gpuCheckError( err, __FILE__, __LINE__ ))
void gpuCheckError(cudaError_t err, const char *file, int line);

struct human_t;
typedef struct human_t human_t;

// A point. Consists of a type and a pointer to a human_t if the type is human
// or end.
typedef struct point_t {
	int type;
	human_t *hum;
} point_t;

typedef struct simple_point_t {
	int x;
	int y;
} simple_point_t;

// A human entity. Consists of a name, a speed, a goal, a shift parameters, and a position
struct human_t {
	char *name;
	int speed;
	int shift;
	point_t *goal;
	int goalX;
	int goalY;
	int posX;
	int posY;
};

#endif /* PATHFINDER_COMMON_H_ */
