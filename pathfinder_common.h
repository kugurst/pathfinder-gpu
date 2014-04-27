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

#define TPATH 1
#define TOBJ 2
#define THUM 3
#define TEND 4

#define HUMCHAR 'S'
#define ENDCHAR 'E'
#define OBCHAR "O"
#define BLKCHAR "B"

struct human_t;
typedef struct human_t human_t;

// A point. Consists of a type and a pointer to a human_t if the type is human
// or end.
typedef struct point_t {
	int type;
	human_t *hum;
} point_t;

// A human entity. Consists of a name, a speed, and a goal
struct human_t {
	char *name;
	int speed;
	int shift;
	point_t *goal;
};

#endif /* PATHFINDER_COMMON_H_ */
