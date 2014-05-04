/*
 * solver.cu
 *
 *  Created on: Apr 27, 2014
 *      Author: mark
 */

#include "cuda.h"

#include <climits>

#include "solver.h"
#include "pathfinder_common.h"
#include "scene.h"

__device__ void addToList(nodeList_t *list, node_t *node) {
	node_t *cur = list->head;
	list->size++;
	if (cur) {
		while (cur->next)
			cur = cur->next;
		cur->next = node;
	} else list->head = node;
}

__device__ node_t *removeCheapest(nodeList_t *list) {
	// If the list is empty, return
	if (list->size == 0) return NULL;

	// Track the cheapest node
	node_t *cheapest = list->head;
	node_t *prevToCheap = NULL;

	// Iteration variables
	node_t *curNode = cheapest->next;
	node_t *prevNode = cheapest;

	while (curNode) {
		if (curNode->f < cheapest->f) {
			cheapest = curNode;
			prevToCheap = prevNode;
		}
		prevNode = curNode;
		curNode = curNode->next;
	}

	// Remove the cheapest node
	if (prevToCheap)
		// We're not at the head of the list
		prevToCheap->next = cheapest->next;
	else
	// Assign the head to be the next node
	list->head = cheapest->next;
	// Invalidate the next pointer, as we're no longer in a list
	cheapest->next = NULL;
	list->size--;
	return cheapest;
}

/* Returns true if the specified coordinates removed a node from this list, or if it was not in the list. False otherwise */
__device__ bool scanList(nodeList_t *list, int x, int y, int cost) {
	// Initialize the current list to the head
	node_t *curNode = list->head;
	// Keep track of the previous list
	node_t *prevNode = NULL;
	// Assume it's not in the list
	bool removed = true;
	// While we haven't reached the end of the list
	while (curNode) {
		// If the node is in the same position
		if (curNode->x == x && curNode->y == y) {
			// Assume that we aren't removing it yet
			removed = false;
			// If the cost of the new node is less than the one in the list, remove the one on the list
			if (cost < curNode->f) {
				// We know we're removing it
				removed = true;
				// If we're at the head of the list, reassign the new head
				if (prevNode == NULL)
					list->head = curNode->next;
				else prevNode->next = curNode->next;
				debugPrintf("curNode->next %p\n", curNode->next);
				// Free the node
				free(curNode);
				list->size--;
				// There will only be one
				break;
			}
		}
		prevNode = curNode;
		curNode = curNode->next;
	}
	return removed;
}

__device__ void freeList(nodeList_t *list) {
	node_t *curNode = list->head;
	node_t *temp = NULL;
	// While we have a valid list
	while (curNode) {
		// Mark the next node
		temp = curNode->next;
		// Free our node
		free(curNode);
		// Go to the next node
		curNode = temp;
	}
}

__device__ void buildPath(nodeList_t *list, node_t *end, node_t *start) {
	// Build the list in reverse
	// Make an identical node to the end
	node_t *curNode = (node_t *) malloc(sizeof(node_t));
	curNode->f = end->f;
	curNode->g = end->g;
	curNode->h = end->h;
	curNode->next = NULL;
	curNode->parent = NULL;
	curNode->x = end->x;
	curNode->y = end->y;
	// stick it on the list
	list->head = curNode;
	list->size = 1;

	// If end and start are the same, we're done
	if (end->x == start->x && end->y == start->y) return;

	// Otherwise
	node_t *prevNode = end->parent;
	while (prevNode) {
		debugPrintf("parent: %d,%d\n", prevNode->x, prevNode->y);
		// Allocate a new node
		node_t *newNode = (node_t *) malloc(sizeof(node_t));
		curNode->parent = newNode;
		newNode->next = curNode;
		newNode->parent = NULL;
		newNode->f = prevNode->f;
		newNode->g = prevNode->g;
		newNode->h = prevNode->h;
		newNode->x = prevNode->x;
		newNode->y = prevNode->y;
		curNode = newNode;
		// The size is incremented
		list->size++;
		// Set the new head
		list->head = curNode;
		// Get the next node
		prevNode = prevNode->parent;
	} debugPrintf("List head coords: %d,%d\n", list->head->x, list->head->y);
}

__device__ nodeList_t *aStar(point_t *grid, human_t *human, int maxWidth,
		int maxHeight) {
	node_t *start = (node_t *) malloc(sizeof(node_t));
	start->parent = NULL;
	start->next = NULL;
	start->x = human->posX;
	start->y = human->posY;
	start->g = 0;
	start->h = abs(start->x - human->goalX) + abs(start->y - human->goalY);
	start->f = start->h;

	nodeList_t *openList = (nodeList_t *) malloc(sizeof(nodeList_t));
	openList->head = start;
	openList->size = 1;
	nodeList_t *closedList = (nodeList_t *) malloc(sizeof(nodeList_t));
	closedList->head = NULL;
	closedList->size = 0;
	nodeList_t *pathList = NULL;
	debugPrintf("initialized\n");

	while (openList->size > 0) {
		node_t *q = removeCheapest(openList);
		debugPrintf("%d,%d | %d\n", q->x, q->y, q->f);

		// Let's see if this was the goal
		if (q->x == human->goalX && q->y == human->goalY) {
			debugPrintf("goal reached: %d,%d\n", q->x, q->y);
			pathList = (nodeList_t *) malloc(sizeof(nodeList_t));
			buildPath(pathList, q, start);
			break;
		}

		int qx = q->x, qy = q->y;
		// For each successor
		for (int yOff = -1; yOff <= 1; yOff++) {
			for (int xOff = -1; xOff <= 1; xOff++) {
				// The center isn't a successor
				if (xOff == 0 && yOff == 0) continue;
				// Make sure it is within bounds
				if (xOff + qx >= maxWidth || yOff + qy >= maxHeight
						|| xOff + qx < 0 || yOff + qy < 0) continue;

				// Get the point
				int px = qx + xOff;
				int py = qy + yOff;
				debugPrintf("point coords: %d,%d\n", px, py);
				point_t point = grid[px + py * maxWidth];

				// If it's not a path or goal, skip it
				// TODO: go through humans
				if (point.type != TPATH && point.type != TEND) continue;

				int gCost = abs(xOff) + abs(yOff) + q->g;
				// We can spend a little more time computing the best path
				int hCost = abs(px - human->goalX) + abs(py - human->goalY);
				int fCost = gCost + hCost;
				debugPrintf("g: %d, h: %d, f: %d\n", gCost, hCost, fCost);
				debugPrintf("scanning list\n");
				bool oRem = scanList(openList, px, py, fCost);
				bool cRem = scanList(closedList, px, py, fCost);
				debugPrintf("removed: %d\n", oRem && cRem);
				// If it was removed from both lists, then add it to open
				if (oRem && cRem) {
					node_t *newNode = (node_t *) malloc(sizeof(node_t));
					newNode->parent = q;
					newNode->next = NULL;
					newNode->x = px;
					newNode->y = py;
					newNode->g = gCost;
					newNode->h = hCost;
					newNode->f = fCost;
					addToList(openList, newNode);
					debugPrintf("added\n");
				} debugPrintf("%d,%d | %d\n", px, py, point.type);
			}
		}
	} debugPrintf("Open list is empty or goal?\n");
	freeList(openList);
	freeList(closedList);

	return pathList;
}

__device__ void printPath(nodeList_t *list) {
	node_t *cur = list->head;
	while (cur) {
		if (cur->next)
			printf("(%d,%d)->", cur->x, cur->y);
		else printf("(%d,%d)G", cur->x, cur->y);
		cur = cur->next;
	}
	printf("\n");
}

__device__ bool moveHuman(point_t *grid, human_t *human, nodeList_t *path,
		int width, int *remainingHumans) {
	// Get our next point
	node_t *node = path->head->next;
	// Get the point on the grid it should be
	point_t *point = &grid[node->x + node->y * width];
	point_t *curPoint = &grid[human->posX + human->posY * width];
	// Try to reserve this point
	bool swapped = false;
	int oldType = atomicCAS(&point->type, TPATH, THUM);
	swapped = (oldType == TPATH);
	// If the point is a goal, that's fine too
	if (!swapped && oldType == TEND)
		swapped = ((oldType = atomicCAS(&point->type, TEND, THUM)) == TEND);
	// If we swapped, then update mark our last position as empty
	if (swapped) {
		atomicCAS(&curPoint->type, THUM, oldType);
		// Update our position
		human->posX = node->x;
		human->posY = node->y;
		// If we have arrived, mark this human as solved
		if (human->posX == human->goalX && human->posY == human->goalY)
			atomicSub(remainingHumans, 1);
	}
	return swapped;
}

__device__ void addToStatsPath(nodeList_t *path, node_t *node) {
	// Allocate a new node for this
	node_t *newNode = (node_t *) malloc(sizeof(node_t));
	// Set its position and cost parameters
	if (node != NULL) {
		newNode->f = node->f;
		newNode->g = node->g;
		newNode->h = node->h;
		newNode->x = node->x;
		newNode->y = node->y;
	} else {
		// Duplicate the last node in this list
		node_t *cur = path->head;
		while (cur->next)
			cur = cur->next;
		newNode->f = cur->f;
		newNode->g = cur->g;
		newNode->h = cur->h;
		newNode->x = cur->x;
		newNode->y = cur->y;
	}
	// Common to both paths
	newNode->next = NULL;
	newNode->parent = NULL;

	// Add it to the list
	if (path->head == NULL) {
		// The list is empty
		path->head = newNode;
	} else {
		// The list is not empty
		node_t *cur = path->head;
		// Skip to the last item
		while (cur->next)
			cur = cur->next;
		// Set the node
		cur->next = newNode;
		newNode->parent = cur;
	}
	path->size++;
}

__device__ void transferPath(stat_t *stats, int id, void *results, int width,
		int height) {
	debugPrintf("%p, %lu\n", results, sizeof(char));
	// Get our offset into the results array, which we'll treat as rows of results
	int resultWidth = width * height * sizeof(simple_point_t) + 2 * sizeof(int);
	debugPrintf("width: %d\n", resultWidth);
	void *rRow = (void *) (((char *) results) + resultWidth * id);
	debugPrintf("base: %p, row: %p, diff: %d, id: %d\n", results, rRow,
			((char * ) rRow) - ((char * ) results), id);
	// Write the stats to the results
	node_t *curNode = stats->path->head;
	// Mark the result we are writing
	unsigned int pos = 0;
	while (curNode) {
		// Get the current result
		simple_point_t *point = ((simple_point_t *) rRow) + pos++;
		// Set is position
		point->x = curNode->x;
		point->y = curNode->y;
		curNode = curNode->next;
	}
	// Go to the end of the row and write how many elements we read and the collision count
	unsigned int *nums = (unsigned int *) (((char *) rRow)
			+ (resultWidth - sizeof(int) * 2));
	debugPrintf("ints: %p, diff: %d\n", nums,
			((char * ) nums) - ((char * ) rRow));
	nums[0] = stats->collisions;
	nums[1] = pos;
	// Finally, free the paths and stats object
	free(stats->path);
	free(stats);
}

__global__ void solveScene(point_t *grid, human_t *humans, stat_t *stats,
		nodeList_t **paths, int maxWidth, int maxHeight, int numHumans,
		int *remainingHumans, void *results) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numHumans) return; // No human to work on

	// Initialize the stats
	if (id < numHumans && stats[id].path == NULL) {
		stats[id].path = (nodeList_t *) malloc(sizeof(nodeList_t));
		stats[id].path->head = NULL;
		stats[id].path->size = 0;
		// Make a start node for the stats path
		node_t node;
		node.parent = NULL;
		node.next = NULL;
		node.x = humans[id].posX;
		node.y = humans[id].posY;
		node.g = 0;
		node.h = (humans[id].posX - humans[id].goalX)
				* (humans[id].posX - humans[id].goalX)
				+ (humans[id].posY - humans[id].goalY)
						* (humans[id].posY - humans[id].goalY);
		node.f = node.h;
		addToStatsPath(stats[id].path, &node);
	}

	// Get our current human
	human_t *hum = &humans[id];
	// If our human has reached the end, return
	if (hum->posX == hum->goalX && hum->posY == hum->goalY) {
		// Free the paths list
		if (paths[id] != NULL) {
			transferPath(&stats[id], id, results, maxWidth, maxHeight);
			freeList(paths[id]);
			paths[id] = NULL;
		}
		return;
	} else {
		nodeList_t *path;
		// Compute a new path or just use the already existing one
		if (paths[id] != NULL) {
			path = paths[id];
		} else {
			path = aStar(grid, hum, maxWidth, maxHeight);
			paths[id] = path;
		}
		// We found a path, so move a human
		if (path != NULL) {
			bool moved = moveHuman(grid, hum, path, maxWidth, remainingHumans);
			// If we moved, add this path to the final path
			if (moved) {
				node_t *nextNode = path->head->next;
				addToStatsPath(stats[id].path, nextNode);
				// head->next is now our head as that's where we moved
				node_t *toFree = path->head;
				path->head = path->head->next;
				free(toFree);
				debugPrintf("new position: %d,%d\n", hum->posX, hum->posY);
			} else {
				// Stall
				stats[id].collisions++;
				node_t *nextNode = path->head;
				addToStatsPath(stats[id].path, nextNode);
				debugPrintf("id %d blocked\n", id);
			}
		} else {
			// Stall
			addToStatsPath(stats[id].path, NULL);
		}
	}
}

