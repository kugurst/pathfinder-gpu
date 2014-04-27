/*
 * pathfinder_common.h
 *
 *  Created on: Apr 26, 2014
 *      Author: mark
 */

#ifndef PATHFINDER_COMMON_H_
#define PATHFINDER_COMMON_H_

#define ENOFILE -2
#define EREADERR -3
#define EBADFORM -4

// A human entity. Consists of a name, a speed, and location in space
typedef struct human_t {
	char *name;
	double speed;
	float x;
	float y;
} human_t;

#endif /* PATHFINDER_COMMON_H_ */
