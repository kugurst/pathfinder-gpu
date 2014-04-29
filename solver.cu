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

__shared__ bool isLastBlockDone;
__device__ unsigned int count1 = 0;
__device__ unsigned int count2 = 0;

__device__ void addToList(nodeList_t *list, node_t *node)
{
	nodeList_t *curList = list;
	curList->size++;
	if (curList->node)
	{
		while (curList->next)
			curList = curList->next;
		nodeList_t *newList = (nodeList_t *) malloc(sizeof(nodeList_t));
		newList->node = node;
		newList->next = NULL;
		curList->next = newList;
	}
	else
		curList->node = node;
}

__device__ node_t *removeCheapest(nodeList_t **list)
{
	node_t *cheapest = (*list)->node;
	nodeList_t *curList = (*list)->next;
	nodeList_t *prevListToCheap = NULL;
	nodeList_t *cheapList = *list;
	nodeList_t *prevList = *list;
	int prevSize = (*list)->size;
	while (curList)
	{
		node_t *curNode = curList->node;
		if (curNode->f < cheapest->f)
		{
			cheapest = curNode;
			prevListToCheap = prevList;
			cheapList = curList;
		}
		prevList = curList;
		curList = curList->next;
	}
	// Free the list
	// If we're freeing the head of the list...
	if (cheapList == *list)
	{
		// Make sure there's a next list
		if (cheapList->next)
		{
			// Set the list to be that list
			*list = cheapList->next;
			// Free this list
			free(cheapList);
		}
		else
			cheapList->node = NULL;
	}
	else
	{
		// Set the next of the previous list to our next
		prevListToCheap->next = cheapList->next;
		// Free ourselves
		free(cheapList);
	}
	(*list)->size = prevSize - 1;
	return cheapest;
}

/* Returns true if the specified coordinates removed a node from this list, or if it was not in the list. False otherwise */
__device__ bool scanList(nodeList_t **list, int x, int y, int cost)
{
	// Initialize the current list to the head
	nodeList_t *curList = *list;
	int prevSize = curList->size;
	// Keep track of the previous list
	nodeList_t *prevList = NULL;
	// Assume it's not in the list
	bool removed = true;
	// While we haven't reached the end of the list
	while (curList)
	{
		node_t *curNode = curList->node;
		// If this list has no node, then the list is empty
		if (curNode == NULL)
			break;
		// If the node is in the same position
		if (curNode->x == x && curNode->y == y)
		{
			// Assume that we aren't removing it yet
			removed = false;
			// If the cost of the new node is less than the one in the list, remove it
			if (cost < curNode->f)
			{
				// We know we're removing it
				removed = true;
				// Free the node
				free(curList->node);
				// If we're at the head of the list, reassign the new head
				if (prevList == NULL)
				{
					nodeList_t *nextList = curList->next;
					// If the next item exists, use that
					if (nextList)
					{
						*list = nextList;
						free(curList);
					}
					else
					{
						// Otherwise, set this list to empty
						curList->node = NULL;
					}
				}
				else
				{
					// We're not at the beginning of the list
					// Assign the previous list's next to our next
					prevList->next = curList->next;
					// We're out of the list, so free us
					free(curList);
				}
				(*list)->size = prevSize - 1;
				// There will only be one
				break;
			}
		}
		prevList = curList;
		curList = curList->next;
	}
	return removed;
}

__device__ void freeList(nodeList_t *list)
{
	nodeList_t *curList = list;
	// While we have a valid list
	while (curList)
	{
		// Free our node
		if (curList->node)
			free(curList->node);
		// Go to the next list
		nodeList_t *nextList = curList->next;
		// Free ourselves
		free(curList);
		curList = nextList;
	}
}

__device__ void buildPath(nodeList_t **list, node_t *end, node_t *start)
{
	// First, malloc space for the path
	*list = (nodeList_t *) malloc(sizeof(nodeList_t));

	// Build the list in reverse
	nodeList_t *afterList = *list;
	// Make an identical node to the end
	node_t *endI = (node_t *) malloc(sizeof(node_t));
	endI->f = end->f;
	endI->g = end->g;
	endI->h = end->h;
	endI->next = NULL;
	endI->parent = NULL;
	endI->x = end->x;
	endI->y = end->y;
	// Set the last element to be the end
	afterList->node = endI;
	afterList->next = NULL;
	afterList->size = 1;

	// If end and start are the same, we're done
	if (end->x == start->x && end->y == start->y)
		return;

	// Otherwise
	node_t *prevNode = end->parent;
	node_t *curNode = endI;
	while (prevNode)
	{
		debugPrintf("parent: %d,%d\n", prevNode->x, prevNode->y);
		// Allocate a new list
		nodeList_t *newHead = (nodeList_t *) malloc(sizeof(nodeList_t));
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
		// The next list is afterList
		newHead->next = afterList;
		// The node is prevNode
		newHead->node = curNode;
		// The size is incremented
		newHead->size = afterList->size + 1;
		// Set the new head
		*list = newHead;
		// This is now the next list to replace
		afterList = newHead;
		// Get the next node
		prevNode = prevNode->parent;
	} debugPrintf("List head coords: %d,%d\n", (*list)->node->x, (*list)->node->y);
}

__device__ nodeList_t *aStar(point_t *grid, human_t *human, int maxWidth,
		int maxHeight)
{
	node_t *start = (node_t *) malloc(sizeof(node_t));
	start->parent = NULL;
	start->next = NULL;
	start->x = human->posX;
	start->y = human->posY;
	start->g = 0;
	start->h = (human->posX - human->goalX) * (human->posX - human->goalX)
			+ (human->posY - human->goalY) * (human->posY - human->goalY);
	start->f = start->g + start->h;

	nodeList_t *openList = (nodeList_t *) malloc(sizeof(nodeList_t));
	openList->node = start;
	openList->next = NULL;
	openList->size = 1;
	nodeList_t *closedList = (nodeList_t *) malloc(sizeof(nodeList_t));
	closedList->node = NULL;
	closedList->next = NULL;
	closedList->size = 0;
	nodeList_t *pathList = NULL;
	debugPrintf("initialized\n");

	while (openList->size > 0)
	{
		node_t *q = removeCheapest(&openList);
		addToList(closedList, q);
		debugPrintf("%d,%d | %d\n", q->x, q->y, q->f);

		// Let's see if this was the goal
		if (q->x == human->goalX && q->y == human->goalY)
		{
			debugPrintf("goal reached: %d,%d\n", q->x, q->y);
			buildPath(&pathList, q, start);
			break;
		}

		int qx = q->x, qy = q->y;
		// For each successor
		for (int yOff = -1; yOff <= 1; yOff++)
		{
			for (int xOff = -1; xOff <= 1; xOff++)
			{
				// The center isn't a successor
				if (xOff == 0 && yOff == 0)
					continue;
				// Make sure it is within bounds
				if (xOff + qx >= maxWidth || yOff + qy >= maxHeight
						|| xOff + qx < 0 || yOff + qy < 0)
					continue;

				// Get the point
				int px = qx + xOff;
				int py = qy + yOff;
				debugPrintf("point coords: %d,%d\n", px, py);
				point_t point = grid[px + py * maxWidth];

				// If it's not a path or goal, skip it
				if (point.type != TPATH && point.type != TEND)
					continue;

				// Since xOff is at most one, xOff ^ 2 is still 1
				int gCost = abs(xOff) + abs(yOff) + q->g;
				int hCost = (px - human->goalX) * (px - human->goalX)
						+ (py - human->goalY) * (py - human->goalY);
				int fCost = gCost + hCost;
				debugPrintf("g: %d, h: %d, f: %d\n", gCost, hCost, fCost);
				bool removed = scanList(&openList, px, py, fCost)
						&& scanList(&closedList, px, py, fCost);
				debugPrintf("removed: %d\n", removed);
				// If it was removed from both lists, then add it to open
				if (removed)
				{
					node_t *newNode = (node_t *) malloc(sizeof(node_t));
					newNode->parent = q;
					newNode->x = px;
					newNode->y = py;
					newNode->g = gCost;
					newNode->h = hCost;
					newNode->f = fCost;
					addToList(openList, newNode);
					debugPrintf("added\n");
				} debugPrintf("%d,%d | %d\n\n", px, py, point.type);
			}
		}
	} debugPrintf("Open list is empty or goal?\n");
	freeList(openList);
	freeList(closedList);

	return pathList;
}

__device__ void syncAllThreads(unsigned int *syncCounter,
		unsigned int *iterations)
{
	__syncthreads();
	if (threadIdx.x == 0)
		atomicInc(syncCounter, gridDim.x - 1);
	volatile unsigned int *counter = syncCounter;
	do
	{
	} while (*counter > 0);
	(*iterations)++;
}

__device__ void printPath(nodeList_t *list)
{
	nodeList_t *cur = list;
	while (cur)
	{
		if (cur->next)
			printf("(%d,%d)->", cur->node->x, cur->node->y);
		else
			printf("(%d,%d)G", cur->node->x, cur->node->y);
		cur = cur->next;
	}
	printf("\n");
}

__device__ bool moveHuman(point_t *grid, human_t *human, nodeList_t *path,
		int width, int *remainingHumans)
{
	// Get our next point
	node_t *node = path->next->node;
	// Get the point on the grid it should be
	point_t *point = &grid[node->x + node->y * width];
	point_t *curPoint = &grid[human->posX + human->posY * width];
	// Try to reserve this point
	bool swapped = false;
	int oldType = atomicCAS(&point->type, TPATH, THUM);
	swapped = oldType == TPATH;
	// If the point is a goal, that's fine too
	if (!swapped && oldType == TEND)
		swapped = atomicCAS(&point->type, TEND, THUM) == TEND;
	// If we swapped, then update mark our last position as empty
	if (swapped)
	{
		atomicCAS(&curPoint->type, THUM, TPATH);
		// Update our position
		human->posX = node->x;
		human->posY = node->y;
		// If we have arrived, mark this human as solved
		if (human->posX == human->goalX && human->posY == human->goalY)
			atomicSub(remainingHumans, 1);
		__threadfence();
	}
	return swapped;
}

__device__ void addToStatsPath(nodeList_t *path, node_t *node)
{
	// Allocate a new node for this
	node_t *newNode = (node_t *) malloc(sizeof(node_t));
	// Set its position and cost parameters
	newNode->f = node->f;
	newNode->g = node->g;
	newNode->h = node->h;
	newNode->x = node->x;
	newNode->y = node->y;
	newNode->next = NULL;
	newNode->parent = NULL;
	// Add it to the list
	if (path->node == NULL)
	{
		// The list is empty
		path->node = newNode;
		return;
	}
	else
	{
		// The list is not empty
		nodeList_t *curList = path;
		// Skip to the last item
		while (curList->next)
			curList = curList->next;
		// Allocate a list
		nodeList_t *newList = (nodeList_t *) malloc(sizeof(nodeList_t));
		// Set the node and list
		newList->node = newNode;
		newList->next = NULL;
		curList->next = newList;
		newNode->parent = curList->node;
	}
	path->size++;
}

__device__ void transferPath(stat_t *stats, int id, void *results, int width,
		int height)
{
	debugPrintf("%p, %lu\n", results, sizeof(char));
	// Get our offset into the results array, which we'll treat as rows of results
	int resultWidth = width * height * sizeof(simple_point_t) + 2 * sizeof(int);
	debugPrintf("width: %d\n", resultWidth);
	void *rRow = (void *) (((char *) results) + resultWidth * id);
	debugPrintf("base: %p, row: %p, diff: %d, id: %d\n", results, rRow, ((char *) rRow) - ((char *) results), id);
	// Write the stats to the results
	nodeList_t *curList = stats->path;
	// Mark the result we are writing
	unsigned int pos = 0;
	while (curList)
	{
		// Get the current result
		simple_point_t *point = ((simple_point_t *) rRow) + pos++;
		// Set is position
		point->x = curList->node->x;
		point->y = curList->node->y;
		curList = curList->next;
	}
	// Go to the end of the row and write how many elements we read and the collision count
	unsigned int *nums = (unsigned int *) (((char *) rRow)
			+ (resultWidth - sizeof(int) * 2));
	debugPrintf("ints: %p, diff: %d\n", nums, ((char *) nums) - ((char *) rRow));
	nums[0] = stats->collisions;
	nums[1] = pos;
	// Finally, free the paths and stats object
	free(stats->path);
	free(stats);
}

__global__ void solveScene(point_t *grid, human_t *humans, stat_t *stats,
		int maxWidth, int maxHeight, int numHumans, int *remainingHumans,
		void *results, unsigned int *itrCnt, unsigned int *iterations)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	bool transferredPath = false;
	debugPrintf("id: %d\n", id);
	iterations[id] = 0;
	// Initialize the stats
	if (id < numHumans)
	{
		stats[id].collisions = 0;
		stats[id].path = (nodeList_t *) malloc(sizeof(nodeList_t));
		stats[id].path->node = NULL;
		stats[id].path->next = NULL;
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
		node.f = node.g + node.h;
		addToStatsPath(stats[id].path, &node);
	}
	while (*remainingHumans != 0)
	{
		// If we don't have a human, return
		if (id >= numHumans)
		{
			// Unless we're the first thread (synchronization purposes)
			if (threadIdx.x != 0)
				return;
			// Sync threads
			syncAllThreads((iterations[id] % 2 == 0) ? &count1 : &count2,
					&iterations[id]);
		}
		else
		{
			// Get our current human
			human_t *hum = &humans[id];
			// If our human has reached the end, return
			if (hum->posX == hum->goalX && hum->posY == hum->goalY)
			{
				if (!transferredPath)
				{
//					printPath(stats[id].path);
					debugPrintf("id2: %d\n", id);
					transferPath(&stats[id], id, results, maxWidth, maxHeight);
					transferredPath = true;
				}
				// Unless we're the first thread (synchronization purposes)
				if (threadIdx.x != 0)
					return;
				// Sync threads
				syncAllThreads((iterations[id] % 2 == 0) ? &count1 : &count2,
						&iterations[id]);
			}
			else
			{
				debugPrintf("%d\n", gridDim.x);
				nodeList_t *path = aStar(grid, hum, maxWidth, maxHeight);
				// We found a path, so move a human
				if (path != NULL)
				{
					bool moved = moveHuman(grid, hum, path, maxWidth,
							remainingHumans);
					// If we moved, add this path to the final path
					if (moved)
					{
						node_t *nextNode = path->next->node;
						addToStatsPath(stats[id].path, nextNode);
						debugPrintf("new position: %d,%d\n", hum->posX, hum->posY);
					}
					else
					{
						// Stall
						stats[id].collisions++;
						node_t *nextNode = path->node;
						addToStatsPath(stats[id].path, nextNode);
						debugPrintf("blocked\n");
					}
					freeList(path);
					debugPrintf("freed path\n");
				}
				else
				{
					// Stall
					node_t node;
					node.parent = NULL;
					node.next = NULL;
					node.x = hum->posX;
					node.y = hum->posY;
					node.g = 0;
					node.h = (hum->posX - hum->goalX) * (hum->posX - hum->goalX)
							+ (hum->posY - hum->goalY)
									* (hum->posY - hum->goalY);
					node.f = node.g + node.h;
					addToStatsPath(stats[id].path, &node);
				}
				// Verify that the scene has been solved
				syncAllThreads((iterations[id] % 2 == 0) ? &count1 : &count2,
						&iterations[id]);
			}
		}
	}
	// The very last threads won't have transfered their humans yet
	if (!transferredPath && id < numHumans)
	{
//		printPath(stats[id].path);
		debugPrintf("id2: %d\n", id);
		transferPath(&stats[id], id, results, maxWidth, maxHeight);
		transferredPath = true;
		*itrCnt = iterations[id];
	}
}

