/*
 * scene.h
 *
 *  Created on: Apr 26, 2014
 *      Author: mark
 */

#ifndef SCENE_H_
#define SCENE_H_

#include <stdio.h>

#include "pathfinder_common.h"

#define TPATH 1
#define TOBJ 2
#define THUM 3

typedef struct point_t {
	int type;
} point_t;

typedef struct scene_t {
	point_t **grid;
} scene_t;

int buildMap(FILE *, scene_t *, int *, int *);


#endif /* SCENE_H_ */
