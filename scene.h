/*
 * scene.h
 *
 *  Created on: Apr 26, 2014
 *      Author: mark
 */

#ifndef SCENE_H_
#define SCENE_H_

#include <cstdio>
#include <string>
#include <map>

#include "pathfinder_common.h"

using namespace std;

typedef struct scene_t {
	point_t **grid;
} scene_t;

int buildMap(FILE *, scene_t *, int *, int *, map<string, human_t *> *, map<string, point_t *> *);


#endif /* SCENE_H_ */
