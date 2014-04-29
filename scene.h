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
	int width;
	int height;
} scene_t;

int buildMap(FILE *, scene_t *, map<string, human_t *> *, map<string, point_t *> *);
int linearizeGrid(scene_t *, point_t **);
void analyzeResults(void *, map<string, human_t *> *, scene_t *, unsigned int, human_t *);
void freeScene(scene_t *, map<string, human_t *> *);


#endif /* SCENE_H_ */
