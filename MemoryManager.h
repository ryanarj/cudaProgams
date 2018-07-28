//
//  MemoryManager.h
//  Final-SIMD-Comparisions
// header including required modules
//  Created by zhila nouri on 10/16/17.
//  Copyright © 2017 zhila nouri. All rights reserved.

/**<************************# Includes ********************************/
#include<stdlib.h>
#include<stdio.h>
#include <stdbool.h>

/**<************************ Structure definition *********************/
typedef int POINTID;
typedef int NODEID;
typedef int BUFFID;
typedef int POLYID;

/**
* The CPU has three main structures, NODE (nodes of the quadtree),
* POINT (the input points for the quadtree) and
* POINT_BUFFER (the array of points which each leaf node of the quadtree holds).
*/
typedef struct POINT {
	double x;
	double y;
	int index;

} POINT;



typedef struct NODE {
	// Level
	unsigned int level;
	// Keep track of type of NODE : LEAF, LINK
	unsigned int type;
	// Location in 2D space
	float posX;
	float posY;
	// Size of quadrant
	float width;
	float height;
	// Description of points
	//int count[4];
	int total;
	int index;
	int parent_index;
	int open;
	NODEID child[4];
	// This handles the 4 regions 0,1,2,3 representing sub-quadrants
	BUFFID pBuffer;
	//long leafBuffer[5];
	// int  leafBufferCount;
	int totalRegisterQuery;
	long leafBufferStart;
	int newCount;
	bool split;
	unsigned int merge;

	POINT points[1024];

} NODE;

/**<***************** enums ******************************/

enum {
	TYPE_NONE = 0, TYPE_ROOT, TYPE_LINK, TYPE_LEAF, TYPE_INV
};

enum {
	FullyOverlapped = 0, PartiallyOverlapped
};

NODE* d_NODE;
float* d_POINT_x;
float* d_POINT_y;
int* d_POINT_id;
NODEID* d_POINT_nodeid;


/**<************************ Memory preallocation **********************/
// Preallocating memory for point structure.