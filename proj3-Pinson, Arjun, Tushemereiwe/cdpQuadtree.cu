/*

	Project 3
	Summer 2018
	Brian Pinson
	Karshan Arjun
	Mark Tushemereiwe 


*/

/** *******************************************************************
*  File name : quadtreeGPU.cu
*  Construct quadtree in CPU. The version with all edited function
*
** *******************************************************************/
/**<************************# Includes ********************************/
#include<stdio.h>
#include<stdlib.h>
#include"MemoryManager.h"
#include<unistd.h>
#include<sys/time.h>
#include <stdbool.h>
#include<stdlib.h>
#include<cstdlib>
#include <cuda.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include<time.h>
#include<string.h>
#include <iostream>
#include <cmath>
#include <limits>
#include <float.h>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#ifdef __CDT_PARSER__

/**<************************# Defines *********************************/
#define __host__
#define __shared__
#define CUDA_KERNEL_DIM(...)
#else
#define CUDA_KERNEL_DIM(...)  <<< __VA_ARGS__ >>>
#endif
#define BUILD_FULL 1
#define BUILD_ADAPTIVE 2
#define MODE_RANDOM 1
#define MODE_FILE 2
#define TRUE 1
#define FALSE 0
#define pMax 32
#ifndef RANGE
#define RANGE 24000
//#define RANGE 1024
#endif

#define BLOCK_SIZE 1024
#define CUDA_BLOCK_SIZE 64
#define STACK_MAX 36
#define BUFFER_SIZE 1024
#define Leaf_SIZE 1024
#define INSERT_BLOCK_SIZE 1024
#define PAGE_SIZE 40
#define NB_PAGE_SIZE 50
#define LEAF_BUFFER_SIZE 1024
#define MAX_LEAF_CAPACITY 5120

__device__ __constant__ int     bucket_size;
__device__ __constant__ int     max_levels = 10;

__constant__ long long PDH_acnt_CUDA; // constant memory number of points
__constant__ double PDH_res_CUDA; // constant memory width size
extern __shared__ double sharedMemory[]; // shared memory to contain points


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


//typedef int POINTID;
//typedef int NODEID;
//typedef int BUFFID;


/* These are for an old way of tracking time */
struct timezone Idunno;
struct timeval startTime, endTime;


/* helps keep track of tree child nodes*/
struct tree_path
{
	NODEID child[4];
};


/*
int numLevels = 10;
int maxNodes=349525;
int maxLeaves=262144;
int maxLeafParent=65536;
//level 8
int maxNodes=21845;
int maxLeaves=16384;
int maxLeafParent=4096;
*/
/**<***************** Global variables ****************************/
int pointMode = MODE_RANDOM;
char *inputPointFileName;
char *outputTreeFileName;
int rangeSize = RANGE;
//int bucketSize = 512;
int bucketSize = 1024;
//int numPoints = 8192000;
//int numPoints = 409600;
int numPoints = 16384000;
int numLevels = 10;
int maxNodes = 349525;
int maxLeaves = 262144;
int maxLeafParent = 65536;
int numSearches = 10;
int printTree = 1;
int outputTree = 0;
int quadTreeMode = BUILD_FULL;
//int quadTreeMode = BUILD_ADAPTIVE;
//int numPolygon = 1099120;
int pointRangeX = RANGE;
int pointRangeY = RANGE;
int completeIndex = 0;
int NotIndex = 0;
int PartialIndex = 0;
int arraysize = 100;
int globalLevel = 0;
int globalpoint = 0;

/**<***************** enums ******************************/
//enum {
//    TYPE_NONE = 0, TYPE_ROOT, TYPE_LINK, TYPE_LEAF, TYPE_INV
//};
//
//enum {
//    FullyOverlapped = 0, PartiallyOverlapped
//};

//for tree construction
int *d_node_counter;
int * d_split_node;
int * d_node_id;
int * d_level;
int* d_point_node;
__device__ unsigned int   d_node_allocate = 0;
__device__ unsigned int   d_point_allocate = 0;
//define constant
//__device__ unsigned int   d_max_level= 0;
unsigned int   h_node_allocate = 0;
unsigned int   h_point_allocate = 0;


struct buffer {
	//int id;
	int leafId;
	int numberOfQueries;
	unsigned long int queries[BUFFER_SIZE];
};


typedef struct LEAF_BUFFER {
	// Array of points
	unsigned long int queryList[LEAF_BUFFER_SIZE];
	//unsigned int querytCount;
	//unsigned long int nextBufferId;
} LEAF_BUFFER;


struct Output {
	unsigned long long int offset[7];
	int page_num;
}Output;



float *d_query_POINT_x;
float *d_query_POINT_y;




int  *d_query_POINT_id;




float2 *d_positions;


unsigned long long int *leaf_m_address;

int* d_POINT_nodeid;


//for output
struct Output *d_output;
struct Output *d_output_nonBuffer;
struct Output *h_output;

__device__ unsigned int   d_leaves_allocate = 0;
__device__ unsigned int   d_leaf_blocks = 0;
int* d_leave_list;
__device__ int   d_zeros = 0;
unsigned int   h_zeros = 0;
//for saving the intersecting leaves
int *d_intersecting_leave_nodes;  //save intersecting leave nodes
int *d_intersecting_leave_count;  //count the intersection

__device__ int   d_counter_one = 0;
unsigned int   h_counter_one = 0;
__device__ int   d_split_array_zero = 0;
unsigned int   h_split_array_zero = 0;

__global__ void setRootNodeKernel(float xPos, float yPos, int *d_node_counter, int *d_split_node, int *d_level, float2 *d_positions, int numberOfPoints) {


	d_node_counter[0] = numberOfPoints;
	d_split_node[0] = 1;
	d_positions[0].x = xPos;
	d_positions[0].y = yPos;
	d_level[0] = 0;

}


//get direction
__device__  int getNodeDirection(float posX, float posY, float width, float height, float x, float y) {

	if ((x >= posX) && (x < posX + width) && (y >= posY + height)
		&& (y < posY + height + height)) {
		return 0;
	}
	else if ((x >= posX + width) && (x < posX + width + width) && (y >= posY + height)
		&& (y < posY + height + height)) {
		return 1;
	}
	else if ((x >= posX) && (x < posX + width) && (y >= posY)
		&& (y < posY + height)) {
		return 2;
	}
	else if ((x >= posX + width) && (x < posX + width + width) && (y >= posY)
		&& (y < posY + height)) {
		return 3;
	}
	else {
		return -1;
	}


}

__global__ void countThePointsInPositions(float width, float height, int level, float* d_queries_x, float* d_queries_y, int *d_node_counter, int *d_split_node, int *d_level, int numberOfthreads, int blocks_num, float2 *d_positions, int *d_point_node) {

	const unsigned long long int  tid = threadIdx.x + (blockIdx.x*blockDim.x);
	if (tid < numberOfthreads) {

		register float x = d_queries_x[tid];
		register float y = d_queries_y[tid];
		register int myCount = 0;
		register int direction = -1;
		register int node_Id = d_point_node[tid];
		register float posX = d_positions[node_Id].x;
		register float posY = d_positions[node_Id].y;
		register int mem_position;

		if (d_split_node[node_Id] == 1) {

			direction = getNodeDirection(posX, posY, width, height, x, y);
			if (direction != -1) {
				mem_position = (((node_Id * 4) + direction) + 1);
				d_point_node[tid] = mem_position;
				//                if (tid ==0){
				//                    printf("x:%f, y: %f , direction:%i, node_id:%i, dir:%i , xpos:%f, ypos:%f \n", x, y, direction, node_Id, mem_position, posX, posY);
				//                }

				if ((d_split_node[mem_position] == 0 || (level == max_levels))) {
					//&& d_split_node[mem_position]==0
					myCount = atomicAdd(&d_node_counter[mem_position], 1);
					if (myCount == bucket_size && (level < max_levels)) {
						d_split_node[mem_position] = 1;
						d_level[mem_position] = level;

						//                        float width = pWidth / 2.00;
						//                        float height = pHeight / 2.00;
						//
						switch (direction) {

						case 0: // NW
							posX = posX;
							posY = posY + height;
							d_positions[mem_position].x = posX;
							d_positions[mem_position].y = posY;

							break;
						case 1: // NE
							posX = posX + width;
							posY = posY + height;
							d_positions[mem_position].x = posX;
							d_positions[mem_position].y = posY;

							break;
						case 2: // SW
							posX = posX;
							posY = posY;
							d_positions[mem_position].x = posX;
							d_positions[mem_position].y = posY;

							break;
						case 3: // SE
							posX = posX + width;
							posY = posY;
							d_positions[mem_position].x = posX;
							d_positions[mem_position].y = posY;

							break;
						}

						// printf("tid: %li, node id:%i, xpos:%f, ypos:%f, dplit:%i\n", tid, mem_position, posX, posY, d_split_node[mem_position]);
					}
				}

			}
		}

	}


	__syncthreads();
}









__device__ inline void device_setNode(NODEID nodeid, float x, float y, float w, float h, int type, int level, int parentIndex, NODE* d_NODE, int open) {
	// Get memory for node.

	// Set the 5 parameters.
	d_NODE[nodeid].index = nodeid;
	d_NODE[nodeid].posX = x;
	d_NODE[nodeid].posY = y;
	d_NODE[nodeid].width = w;
	d_NODE[nodeid].height = h;
	d_NODE[nodeid].level = level;
	// Reset all of the tracking values.
	int i;
	for (i = 0; i < 4; i++)
	{
		d_NODE[nodeid].child[i] = -1;
		//node->count[i] = 0;
	}
	d_NODE[nodeid].total = 0;
	//node->index = 0;
	//node->offset = 0;
	d_NODE[nodeid].open = open;
	d_NODE[nodeid].type = type;
	d_NODE[nodeid].pBuffer = -1;
	d_NODE[nodeid].parent_index = parentIndex;
	d_NODE[nodeid].leafBufferStart = -1;
	d_NODE[nodeid].totalRegisterQuery = 0;
	//d_NODE[nodeid].newCount=0;


}

__device__ inline int getDirection(unsigned long long int tid) {

	int direction = (tid % 4);
	int actualDirection;

	switch (direction) {
	case 0:
		//child SE dir =3
		actualDirection = 3;
		break;
	case 1:
		//child NW dir =0
		actualDirection = 0;
		break;
	case 2:
		//child NE dir=1
		actualDirection = 1;
		break;
	case 3:
		//child SW dir =2
		actualDirection = 2;
		break;
	}


	return actualDirection;

}

__global__ void createRootNodeKernel(float posX, float posY, float pWidth, float pHeight, struct NODE* d_NODE, int *d_node_id) {
	register int myindex = 0;
	myindex = atomicAdd(&d_node_allocate, 1);
	d_node_id[0] = myindex;
	device_setNode(myindex, posX, posY, pWidth, pHeight, TYPE_ROOT, 0, -1, d_NODE, false);


}


__global__ void createParentNodesKernel(float posX, float posY, float pWidth, float pHeight, struct NODE* d_NODE, int *d_node_counter, int *d_split_node, int maxNodes, int *d_node_id, int *d_level, float2 *d_positions) {

	const unsigned long long int  tid = threadIdx.x + (blockIdx.x*blockDim.x);
	if (tid < maxNodes && d_node_counter[tid] != 0 && tid != 0) {
		register int myindex = 0;
		myindex = atomicAdd(&d_node_allocate, 1);
		d_node_id[tid] = myindex;
		//        if (tid == 0){
		//
		//            device_setNode(myindex, posX, posY, pWidth, pHeight, TYPE_ROOT, 0, 0, d_NODE, false);
		//            //printf("my index is:%i  \n", myindex);
		//        }
		//        else {

		register int direction = getDirection(tid);
		register int parent;
		parent = (tid - direction - 1) / 4;
		register int level;
		register float xPos;
		register float yPos;
		register int type;
		register float width;
		register float height;
		register int open;
		// register int total;
		if (d_split_node[tid] == 1) {
			//this is a link node
			level = d_level[tid];
			xPos = d_positions[tid].x;
			yPos = d_positions[tid].y;
			type = TYPE_LINK;
			width = pWidth / (float)(pow((float)2, (float)level));
			height = pHeight / (float)(pow((float)2, (float)level));
			open = FALSE;
			// total=  d_node_counter[tid];
		}
		else {
			//this is a leaf  node
			level = d_level[parent] + 1;
			type = TYPE_LEAF;
			xPos = d_positions[parent].x;
			yPos = d_positions[parent].y;
			width = pWidth / (float)(pow((float)2, (float)level));
			height = pHeight / (float)(pow((float)2, (float)level));
			open = TRUE;
			// total =0;
			switch (direction) {
			case 0:
				//child SE
				xPos = xPos;
				yPos = yPos + height;
				break;
			case 1:
				//child NW
				xPos = xPos + width;
				yPos = yPos + height;
				break;
			case 2:
				//child NE
				xPos = xPos;
				yPos = yPos;
				break;
			case 3:
				//child SW
				xPos = xPos + width;
				yPos = yPos;
				break;
			}

		}
		//            if (tid==1 ){
		//                printf("my index is:%i , direction is: %i , parent is:%i, total:%i, open:%i, level:%i, xpos:%f, ypos:%f, width:%f, height:%f, type:%i \n", myindex, direction , parent,  d_node_counter[tid], open, level, xPos, yPos, width, height, type );
		//            }


		device_setNode(myindex, xPos, yPos, width, height, type, level, parent, d_NODE, open);



		// }
	}
	__syncthreads();
}



//__global__ void finalNodesSetUpKernel( struct NODE* d_NODE,  int *d_node_counter, int *d_split_node, int maxNodes, int *d_node_id, int *d_leave_list, unsigned long long int *Address, unsigned long long int *d_leaf_buffer_list){
__global__ void finalNodesSetUpKernel(struct NODE* d_NODE, int *d_node_counter, int *d_split_node, int maxNodes, int *d_node_id, int *d_leave_list, unsigned long long int *Address) {

	const unsigned long long int  tid = threadIdx.x + (blockIdx.x*blockDim.x);
	if (tid < maxNodes) {

		if (d_node_counter[tid] != 0 && tid != 0) {

			register int nodeid = d_node_id[tid];
			register int parentNodeId = d_node_id[d_NODE[nodeid].parent_index];
			d_NODE[nodeid].parent_index = parentNodeId;
			register int direction = getDirection(tid);
			d_NODE[parentNodeId].child[direction] = nodeid;

			if (d_split_node[tid] != 1) {
				//this is a leaf
				register int myindex = 0;
				myindex = atomicAdd(&d_point_allocate, d_node_counter[tid]);
				d_NODE[nodeid].pBuffer = myindex;
				myindex = atomicAdd(&d_leaves_allocate, 1);
				d_leave_list[myindex] = nodeid;
				unsigned long long int offsetAddress = atomicAdd(Address, BUFFER_SIZE);
				d_NODE[nodeid].leafBufferStart = offsetAddress;
				d_NODE[nodeid].totalRegisterQuery = 0;


			}

		}
	}

	__syncthreads();
}



//__device__  NODEID  findQuadTreeNodeCuda(NODEID nParentid,  float x, float y,  NODE* d_NODE, unsigned long long int  tid ) {
__device__  NODEID  findQuadTreeNodeCuda(NODEID nParentid, float x, float y, NODE* d_NODE) {
	register float posX, posY;
	register int index;
	if (nParentid == -1)
		return nParentid;

	register NODE nParent = d_NODE[nParentid];
	if (nParent.type == TYPE_LEAF)
		return nParentid;
	// Get the point.
	// Child width and height
	register float width;
	register float height;



	//    if (tid ==0){
	//        printf("nparent is: %i , with:%f, height:%f, child0:%i, child1:%i, child2:%i, child3:%i\n", nParentid, nParent.width , nParent.height,   nParent.child[0],  nParent.child[1],  nParent.child[2],  nParent.child[3]);
	//    }



	while (nParent.type != TYPE_LEAF) {

		width = nParent.width / 2.00;
		height = nParent.height / 2.00;

		for (index = 0; index < 4; index++) {
			switch (index) {
			case 0: // NW
				posX = nParent.posX;
				posY = nParent.posY + height;
				if ((x >= posX) && (x < posX + width) && (y >= posY)
					&& (y < posY + height)) {
					nParentid = nParent.child[0];

				}
				break;
			case 1: // NE
				posX = nParent.posX + width;
				posY = nParent.posY + height;
				if ((x >= posX) && (x < posX + width) && (y >= posY)
					&& (y < posY + height)) {
					nParentid = nParent.child[1];
				}
				break;
			case 2: // SW
				posX = nParent.posX;
				posY = nParent.posY;
				if ((x >= posX) && (x < posX + width) && (y >= posY)
					&& (y < posY + height)) {
					nParentid = nParent.child[2];

				}
				break;
			case 3: // SE
				posX = nParent.posX + width;
				posY = nParent.posY;
				if ((x >= posX) && (x < posX + width) && (y >= posY)
					&& (y < posY + height)) {
					nParentid = nParent.child[3];

				}
				break;
			}
		}
		if (nParentid == -1)
			return nParentid;
		nParent = d_NODE[nParentid];
		//        if (tid ==0){
		//            printf("nparent is: %i \n", nParentid);
		//        }
	}
	return nParentid;
}


__global__ void insertIntoLeafNodes(int *d_node_id, float* d_query_POINT_x, float* d_query_POINT_y, int *d_query_POINT_id, NODE* d_NODE, float *d_POINT_x, float *d_POINT_y, int *d_POINT_id, int *d_point_node, int numPoints, NODEID *d_POINT_nodeid) {

	const unsigned long long int  tid = threadIdx.x + (blockIdx.x*blockDim.x);
	if (tid < numPoints) {
		register int  myindex;
		register NODEID leaf = d_node_id[d_point_node[tid]];
		register float x = d_query_POINT_x[tid];
		register float y = d_query_POINT_y[tid];
		register int index = d_query_POINT_id[tid];

		if (d_NODE[leaf].type == TYPE_LEAF) {
			myindex = atomicAdd(&d_NODE[leaf].total, 1);
			if ((myindex <bucket_size &&  d_NODE[leaf].pBuffer != -1) || (myindex >= bucket_size && d_NODE[leaf].level == max_levels) && d_NODE[leaf].pBuffer != -1) {
				d_POINT_id[(d_NODE[leaf].pBuffer + myindex)] = index;
				d_POINT_x[(d_NODE[leaf].pBuffer + myindex)] = x;
				d_POINT_y[(d_NODE[leaf].pBuffer + myindex)] = y;
				d_POINT_nodeid[(d_NODE[leaf].pBuffer + myindex)] = leaf;

			}
		}
	}

	__syncthreads();

}

/***************************************** end of building the tree ***************************/

/*
search on GPU
*/




//non Buffer range search


__global__ void countTheNumberOfZeros(int *d_split_node, int startLevelNode, int numberOfActiveThreads) {

	const unsigned long long int  tid = threadIdx.x + (blockIdx.x*blockDim.x);

	if (tid < numberOfActiveThreads) {
		if (d_split_node[startLevelNode + tid] == 0) {
			atomicAdd(&d_zeros, 1);
		}

	}

	__syncthreads();

}

//count the number of non-empty nodes in the tree
__global__ void countTheOnesInCounterArray(int *d_node_counter, int maxNodes) {

	const unsigned long long int  tid = threadIdx.x + (blockIdx.x*blockDim.x);

	if (tid < maxNodes) {
		if (d_node_counter[tid] != 0) {
			atomicAdd(&d_counter_one, 1);
		}

	}

	__syncthreads();

}

//count the number of non link nodes in the tree

__global__ void countTheNonLeafNodes(int *d_split_node, int maxNodes) {

	const unsigned long long int  tid = threadIdx.x + (blockIdx.x*blockDim.x);

	if (tid < maxNodes) {
		if (d_split_node[tid] == 0) {
			atomicAdd(&d_split_array_zero, 1);
		}

	}

	__syncthreads();

}

/*

Rebuilds the Quadtree to make it work

*/
__global__ void CUDA_RebuildTree(NODE * d_NODE, int num_of_nodes, tree_path *tree)
{

	int i = 0;
	for (i = 0; i < num_of_nodes; i++)
	{
		int j = 0;
		//		printf("node %i", d_NODE[i].index);
		for (j = 0; j < 4; j++)
		{
			tree[i].child[j] = d_NODE[i].child[j];
			//			printf(" child %i", tree[i].child[j]);
		}
		//		printf(" parent %i\n", d_NODE[i].parent_index);

	}
}

/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

CPU Calculator


&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/


/* descriptors for single atom in the tree */
typedef struct atomdesc
{
	double x_pos;
	double y_pos;
} atom;

typedef struct hist_entry
{
	long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * CPU_histogram;		/* list of all buckets in the histogram   */
bucket * GPU_histogram;		/* list of all buckets in the histogram   */
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */

						/* These are for an old way of tracking time */



						/*
						distance of two points in the atom_list
						*/
double p2p_distance(int ind1, int ind2) {

	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;

	return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}


/*
brute-force SDH solution in a single CPU thread
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;

	for (i = 0; i < PDH_acnt; i++) {
		for (j = i + 1; j < PDH_acnt; j++) {
			dist = p2p_distance(i, j);
			h_pos = (int)(dist / PDH_res);
			CPU_histogram[h_pos].d_cnt++;
		}
	}
	return 0;
}


/*
set a checkpoint and show the (natural) running time in seconds
*/


double report_running_time()
{
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff = endTime.tv_usec - startTime.tv_usec;
	if (usec_diff < 0) {
		sec_diff--;
		usec_diff += 1000000;
	}
	printf("\n\nRunning time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff / 1000000.0);
}


/*
print the counts in all buckets of the histogram
*/
void output_histogram(bucket * input_histogram)
{
	int i;
	long long total_cnt = 0;
	for (i = 0; i< num_buckets; i++) {
		if (i % 5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", input_histogram[i].d_cnt);
		total_cnt += input_histogram[i].d_cnt;
		/* we also want to make sure the total distance count is correct */
		if (i == num_buckets - 1)
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}


/* Prints difference between buckets by using an altered histogram printing function */
void histogram_comparison(bucket *input1, bucket *input2)
{
	printf("Difference Between CPU and CUDA histograms: \n");
	int i;
	long long total_cnt = 0;
	for (i = 0; i< num_buckets; i++) {
		if (i % 5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", input2[i].d_cnt - input1[i].d_cnt);
		total_cnt += input1[i].d_cnt;
		/* we also want to make sure the total distance count is correct */
		if (i == num_buckets - 1)
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}


/* returns distance */
__device__ double CUDA_distance_calculator(double x1, double y1, double x2, double y2)
{
	return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

//////////////////////////////////////////////////
//////////////////////////////////////////////////
/* Quad Tree Traversal							*/
/* Histogram Calculator                         */
//////////////////////////////////////////////////
//////////////////////////////////////////////////


__global__ void CUDA_Calculate_Histogram(bucket *histogram_cuda, NODE *d_NODE, int num_of_nodes, float *d_POINT_x, float *d_POINT_y, NODEID *d_POINT_nodeid, tree_path *tree, int bucket_num)
{
	const unsigned long long int  tid = threadIdx.x + (blockIdx.x*blockDim.x);
	if (tid < PDH_acnt_CUDA)
	{

		NODEID no = d_POINT_nodeid[tid];	// node id
		register double x = d_POINT_x[tid];	// x coordinate
		register double y = d_POINT_y[tid]; // y coordinate


		NODE node = d_NODE[no];				// sets node


		double distance = 0;
		int h_pos = 0;

		int *SHMOut = (int *)sharedMemory;
		int i = 0;
		int j = 0;


		if(threadIdx.x == 0)
			for (; i < bucket_num; i++) SHMOut[i] = 0;

		__syncthreads();


		for (i = tid - node.pBuffer + 1; i < node.total; i++) // scans through current node, finds point's index and calculates histogram for all points of a higher index
		{
			distance = CUDA_distance_calculator(x, y, d_POINT_x[node.pBuffer + i], d_POINT_y[node.pBuffer + i]);
			h_pos = (int)(distance / PDH_res_CUDA);
			atomicAdd(&SHMOut[h_pos], 1);
		}

		for (i = no + 1; i < num_of_nodes; i++) // scans through all nodes greater than current node
		{
			node = d_NODE[i];
			for (j = 0; j < node.total; j++) // calculates histogram for all other points
			{

				distance = CUDA_distance_calculator(x, y, d_POINT_x[node.pBuffer + j], d_POINT_y[node.pBuffer + j]);
				h_pos = (int)(distance / PDH_res_CUDA);
				atomicAdd(&SHMOut[h_pos], 1);
			}
		}

		__syncthreads();

		if (threadIdx.x == 0)
		{
			for (i = 0; i < bucket_num; i++)
			{
				atomicAdd((unsigned long long int*) &histogram_cuda[i].d_cnt, (unsigned long long int) SHMOut[i]);
			}
		}
	}
}


//////////////////////////////////////////////////
//////////////////////////////////////////////////
/* Quad Tree Kernel  							*/
/* Sets up and Launches the CUDA kernel         */
//////////////////////////////////////////////////
//////////////////////////////////////////////////


void Quad_Tree_Traversal(int num_buckets, int grid, int threads, int gpu_nodes, tree_path *tree)
{
	bucket *cuda_histogram = NULL; /* Mallocs histogram in GPU */
	cudaMalloc((void **)&cuda_histogram, num_buckets * sizeof(bucket));
	cudaMemcpy(cuda_histogram, GPU_histogram, num_buckets * sizeof(bucket), cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(PDH_acnt_CUDA, &PDH_acnt, sizeof(signed long long)); // constant memory atom size
	cudaMemcpyToSymbol(PDH_res_CUDA, &PDH_res, sizeof(double)); // constant memory width sizes


	float elapsedTime = 0;

	cudaEvent_t start_time, stop_time;
	cudaEventCreate(&start_time);
	cudaEventCreate(&stop_time);
	cudaEventRecord(start_time, 0);


	CUDA_Calculate_Histogram <<< grid, threads,num_buckets  * sizeof(int)>>> (cuda_histogram, d_NODE, gpu_nodes, d_POINT_x, d_POINT_y, d_POINT_nodeid, tree, num_buckets);


	cudaEventRecord(stop_time, 0);
	cudaEventSynchronize(stop_time);

	cudaEventElapsedTime(&elapsedTime, start_time, stop_time);

	cudaEventDestroy(start_time);
	cudaEventDestroy(stop_time);

	cudaMemcpy(GPU_histogram, cuda_histogram, num_buckets * sizeof(bucket), cudaMemcpyDeviceToHost);
	cudaFree(cuda_histogram);
	printf("\nCUDA Kernel results:\n");
	printf("Time to generate: %0.5f ms\n\n", elapsedTime);
	output_histogram(GPU_histogram);
	printf("\n");

	histogram_comparison(CPU_histogram, GPU_histogram);
	free(GPU_histogram);
}
/////////////////////////
/////////////////////////
/* End Quad Tree traversal */
/////////////////////////
/////////////////////////







/**<************************ Main function ***************************/
/**
* Two techniques to build QuadTrees
* 1- full : extend all the way down, only leafs hold points
*         : counts are kept at intermediate levels
*         : nulls are still used to know where points are.
* 2- adaptive : items are pushed around as needed to form tree
*         : points of LIMIT pushed down.
** ******************************************************************/
int main(int argc, char **argv) {

	if (argc < 4)
	{
		printf("you should insert the number of points, mmaximum number of points alowed in each node, and maximum number of levels alowed in the tree to the program to run\n");
		return 1;
	}
	//number of points in the tree
	unsigned long long int numberOfthreads = atoi(argv[1]);
	numPoints = numberOfthreads;
	//mmaximum number of points alowed in each node
	bucketSize = atoi(argv[2]);
	//maximum number of levels alowed in the tree
	numLevels = atoi(argv[3]);
	//maximum number of possible nodes based on the numLevels
	maxNodes = ((pow(4, numLevels)) - 1) / 3;
	printf("maxNodes is:%i \n", maxNodes);

	PDH_res = 500;


	//  unsigned long long int numberOfthreads = numPoints;
	cudaError_t err = cudaSetDevice(0);


	float *h_POINT_x = (float *)malloc(numPoints * sizeof(float));
	float *h_POINT_y = (float *)malloc(numPoints * sizeof(float));
	int  *h_POINT_id = (int *)malloc(numPoints * sizeof(int));


	memset(h_POINT_x, 0, numPoints * sizeof(float));
	memset(h_POINT_y, 0, numPoints * sizeof(float));
	memset(h_POINT_id, 0, numPoints * sizeof(int));

	atom_list = (atom *)malloc(sizeof(atom)*(numPoints));

	long q;
	srand(time(NULL));
	//srand48(4);
	for (q = 0; q<numPoints; q++) {
		h_POINT_id[q] = q;

		float x = ((float)(rand()) / RAND_MAX) * RANGE;
		float y = ((float)(rand()) / RAND_MAX) * RANGE;

		h_POINT_x[q] = x;
		h_POINT_y[q] = y;
		atom_list[q].x_pos = (double)x;
		atom_list[q].y_pos = (double)y;

	}



	printf("start main \n");


	//NODEID rootNode;


	// Get memory for root node.
	// Start node : root
	//setNode(rootNode, 0, 0, rangeSize, rangeSize, TYPE_ROOT, 0, -1);
	// Create the quadtree.




	//srand48(4);



	// Preallocate memory for all objects in CPU.


	cudaMalloc((void**)&d_node_counter, sizeof(int)*maxNodes);
	gpuErrchk(cudaPeekAtLastError());
	cudaMalloc((void**)&d_split_node, sizeof(int)*maxNodes);
	gpuErrchk(cudaPeekAtLastError());
	cudaMalloc((void**)&d_node_id, sizeof(int)*maxNodes);
	gpuErrchk(cudaPeekAtLastError());
	cudaMalloc((void**)&d_level, sizeof(int)*maxNodes);


	cudaMalloc((void**)&d_positions, sizeof(float2)*maxNodes);
	cudaMalloc((void**)&d_query_POINT_x, sizeof(float)*numPoints);
	cudaMalloc((void**)&d_query_POINT_y, sizeof(float)*numPoints);
	cudaMalloc((void**)&d_point_node, sizeof(int)*numPoints);
	cudaMalloc((void**)&d_query_POINT_id, sizeof(int)*numPoints);

	gpuErrchk(cudaPeekAtLastError());


	cudaMemset(d_node_counter, 0, sizeof(int)*maxNodes);
	cudaMemset(d_split_node, 0, sizeof(int)*maxNodes);
	cudaMemset(d_node_id, 0, sizeof(int)*maxNodes);
	cudaMemset(d_level, 0, sizeof(int)*maxNodes);
	cudaMemset(d_query_POINT_x, 0, sizeof(float)*numPoints);
	cudaMemset(d_query_POINT_y, 0, sizeof(float)*numPoints);
	cudaMemset(d_point_node, 0, sizeof(int)*numPoints);
	cudaMemset(d_query_POINT_id, 0, sizeof(int)*numPoints);

	cudaMemset(d_positions, 0, sizeof(float2)*maxNodes);
	gpuErrchk(cudaPeekAtLastError());
	// cudaMemcpyToSymbol(bucket_size, &bucketSize, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(bucket_size, &bucketSize, sizeof(int));
	gpuErrchk(cudaPeekAtLastError());
	cudaMemcpy(d_query_POINT_x, h_POINT_x, sizeof(float)*numPoints, cudaMemcpyHostToDevice);
	cudaMemcpy(d_query_POINT_y, h_POINT_y, sizeof(float)*numPoints, cudaMemcpyHostToDevice);
	cudaMemcpy(d_query_POINT_id, h_POINT_id, sizeof(int)*numPoints, cudaMemcpyHostToDevice);
	gpuErrchk(cudaPeekAtLastError());
	//thrust
	thrust::device_ptr<int> dev_ptr(d_node_counter);
	thrust::device_ptr<int> dev_split(d_split_node);

	/* start counting time */
	cudaEvent_t start, stop, start1, stop1;
	//run the simulation
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);

	//    err = cudaMemPrefetchAsync(d_query_POINT_x,  sizeof(float)*numPoints, 0);
	//    err = cudaMemPrefetchAsync(d_query_POINT_y,  sizeof(float)*numPoints, 0);
	//    err = cudaMemPrefetchAsync(d_query_POINT_id,  sizeof(int)*numPoints, 0);
	//    err = cudaMemPrefetchAsync(d_node_counter, sizeof(int)*maxNodes  , 0);
	//    err = cudaMemPrefetchAsync(d_split_node, sizeof(int)*maxNodes  , 0);
	//    err = cudaMemPrefetchAsync(d_node_id, sizeof(int)*maxNodes  , 0);
	//    err = cudaMemPrefetchAsync(d_level, sizeof(int)*maxNodes  , 0);
	//    err = cudaMemPrefetchAsync(d_positions, sizeof(float2)*maxNodes  , 0);
	//    err = cudaMemPrefetchAsync(d_point_node, sizeof(int)*numPoints  , 0);

	gpuErrchk(cudaPeekAtLastError());

	float elapsedTime;
	float totalBuildingTime = 0.0;
	//int  blocks_num = 2048000 / BLOCK_SIZE;
	int  blocks_num;
	if (numPoints % BLOCK_SIZE == 0) {
		blocks_num = numPoints / BLOCK_SIZE;
	}
	else {
		blocks_num = numPoints / BLOCK_SIZE + 1;
	}

	int cuda_block_num;
	if (numPoints % CUDA_BLOCK_SIZE == 0) {
		cuda_block_num = numPoints / CUDA_BLOCK_SIZE;
	}
	else {
		cuda_block_num = numPoints / CUDA_BLOCK_SIZE + 1;
	}

	printf("block num is: %i and cuda block num is:%i\n", blocks_num, cuda_block_num);
	printf("BLOCK_SIZE is: %i \n", BLOCK_SIZE);
	dim3 grid(blocks_num, 1, 1);
	dim3 threads(BLOCK_SIZE, 1, 1);
	printf("before calling the first kernel\n");
	// printf("data point in gpu is %p \n", d_POINT+0);
	//todo check the seed


	//todo comment
	//setup_kernel << <1, BLOCK_SIZE >> >(state, unsigned(time(NULL)) +1);
	// gpuErrchk(cudaDeviceSynchronize());

	// Size of quadrant

	float sqrange = RANGE;
	printf("sqrange is: %f \n", sqrange);
	// call the setNode 0

	dim3 grid0(1, 1, 1);
	dim3 threads0(1, 1, 1);
	cudaEventRecord(start1, 0);
	setRootNodeKernel << <grid0, threads0 >> > (0.0, 0.0, d_node_counter, d_split_node, d_level, d_positions, numPoints);
	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&elapsedTime, start1, stop1);
	printf("******** Total Running Time of creating root= %0.5f ms \n", elapsedTime);
	totalBuildingTime = totalBuildingTime + elapsedTime;
	gpuErrchk(cudaDeviceSynchronize());

	int level = 0;
	int max_level = 10;
	bool flag = true;
	//    int startLevelNode = pow (4, level);
	//    int endtLevelNode = pow (4, (level+1));
	int startLevelNode = 1;
	int endtLevelNode = 4;
	int split = 0;
	float width = float(RANGE) / 2.00;
	float height = float(RANGE) / 2.00;

	//for new function
	int numberOfActiveThreads = 0;
	int zeroCount_block_num;
	dim3 threadsz(CUDA_BLOCK_SIZE, 1, 1);
	int previousSplit = 0;

	cudaEventRecord(start1, 0);
	while (level < max_level && flag == true) {
		countThePointsInPositions << <grid, threads >> > (width, height, level + 1, d_query_POINT_x, d_query_POINT_y, d_node_counter, d_split_node, d_level, numPoints, cuda_block_num, d_positions, d_point_node);
		gpuErrchk(cudaDeviceSynchronize());

		numberOfActiveThreads = endtLevelNode - startLevelNode + 1;
		//printf("number of active threads is: %i \n ", numberOfActiveThreads);
		if (numberOfActiveThreads % CUDA_BLOCK_SIZE == 0) {
			zeroCount_block_num = numberOfActiveThreads / CUDA_BLOCK_SIZE;
		}
		else {
			zeroCount_block_num = numberOfActiveThreads / CUDA_BLOCK_SIZE + 1;
		}
		dim3 gridz(zeroCount_block_num, 1, 1);
		// cudaMemcpyToSymbol("d_zeros", &h_zeros, sizeof(int));
		countTheNumberOfZeros << <gridz, threadsz >> > (d_split_node, startLevelNode, numberOfActiveThreads);
		gpuErrchk(cudaDeviceSynchronize());

		cudaMemcpyFromSymbol(&split, d_zeros, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);

		split = split - previousSplit;
		// printf("level is: %i, start is:%i, end is: %i , split:%i \n", level, startLevelNode,endtLevelNode,  split );
		if (split == endtLevelNode - startLevelNode + 1) {
			flag = false;
		}
		else {
			flag = true;
			width = width / 2.00;
			height = height / 2.00;
			level = level + 1;
			startLevelNode = startLevelNode + pow(4, level);
			endtLevelNode = startLevelNode + pow(4, (level + 1)) - 1;
			previousSplit = previousSplit + split;

		}



	}

	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	printf("after calling the kernel\n");
	cudaEventElapsedTime(&elapsedTime, start1, stop1);
	printf("******** Total Running Time of inserting positions is = %0.5f ms \n", elapsedTime);
	totalBuildingTime = totalBuildingTime + elapsedTime;
	printf("level after kernel is: %i \n", level);

	int  node_blocks_num;
	int NODE_BLOCK_SIZE = 32;
	if (maxNodes % NODE_BLOCK_SIZE == 0) {
		node_blocks_num = maxNodes / NODE_BLOCK_SIZE;
	}
	else {
		node_blocks_num = maxNodes / NODE_BLOCK_SIZE + 1;
	}
	printf("maxNode is: %i, node block num is: %i \n", maxNodes, node_blocks_num);
	dim3 grid1(node_blocks_num, 1, 1);
	dim3 threads1(NODE_BLOCK_SIZE, 1, 1);

	cudaEventRecord(start, 0);
	countTheOnesInCounterArray << <grid1, threads1 >> > (d_node_counter, maxNodes);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	gpuErrchk(cudaDeviceSynchronize());
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("******** cont the number of ones in counter array = %0.5f ms \n", elapsedTime);
	cudaMemcpyFromSymbol(&h_counter_one, d_counter_one, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
	printf("ones are: %i non empty nodes \n", h_counter_one);
	int gpu_nodes = h_counter_one;
	totalBuildingTime = totalBuildingTime + elapsedTime;

	cudaEventRecord(start, 0);
	countTheNonLeafNodes << <grid1, threads1 >> > (d_split_node, maxNodes);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	gpuErrchk(cudaDeviceSynchronize());
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("******** cont the number of zeros in split array = %0.5f ms \n", elapsedTime);
	cudaMemcpyFromSymbol(&h_split_array_zero, d_split_array_zero, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
	printf("h_split_zero:%i and non_split_zero :%i and leaves: %i\n", h_split_array_zero, (maxNodes - h_split_array_zero), (gpu_nodes - (maxNodes - h_split_array_zero)));
	totalBuildingTime = totalBuildingTime + elapsedTime;
	split = h_split_array_zero;
	int non_split_zero = maxNodes - split;
	//total number of leaves
	int leaves = gpu_nodes - non_split_zero;
	printf("number os split is: %i and number of leaves is: %i\n", non_split_zero, leaves);

	//    int sum = 0;
	//    sum= thrust::count(thrust::device, dev_ptr, (dev_ptr + maxNodes) , 0);
	//    //total number of nodes
	//    int gpu_nodes= maxNodes - sum ;
	//    printf("zero is:%i and nodes is: %i \n", sum, gpu_nodes);
	//
	//    split= thrust::count(thrust::device, dev_split, (dev_split + maxNodes) , 0);
	//    //non leaf nodes
	//    int non_split_zero= maxNodes - split;
	//    //total number of leaves
	//    int leaves = gpu_nodes - non_split_zero;
	//    printf("number os split is: %i and number of leaves is: %i\n", non_split_zero, leaves);


	//    int numberOfNodes;
	cudaMalloc((void**)&d_NODE, sizeof(NODE)*gpu_nodes);
	cudaMemset(d_NODE, 0, sizeof(NODE)*gpu_nodes);
	cudaMalloc((void**)&d_POINT_x, sizeof(float)*numPoints);
	cudaMalloc((void**)&d_POINT_y, sizeof(float)*numPoints);
	cudaMalloc((void**)&d_POINT_nodeid, sizeof(NODEID)*numPoints);
	cudaMemset(d_POINT_nodeid, 0, sizeof(NODEID)*numPoints);
	cudaMemset(d_POINT_x, 0, sizeof(float)*numPoints);
	cudaMemset(d_POINT_y, 0, sizeof(float)*numPoints);
	cudaMalloc((void**)&d_POINT_id, sizeof(int)*numPoints);
	cudaMemset(d_POINT_id, 0, sizeof(int)*numPoints);
	cudaMalloc((void**)&d_leave_list, sizeof(int)*leaves);
	cudaMemset(d_leave_list, 0, sizeof(int)*leaves);
	cudaMalloc((void**)&leaf_m_address, sizeof(unsigned long long int));
	cudaMemset(leaf_m_address, 0, sizeof(unsigned long long int));

	gpuErrchk(cudaPeekAtLastError());

	//create root Node
	cudaEventRecord(start, 0);
	createRootNodeKernel << <grid0, threads0 >> > (0.0, 0.0, sqrange, sqrange, d_NODE, d_node_id);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	gpuErrchk(cudaDeviceSynchronize());
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("******** Total Running Time of creating root node = %0.5f ms \n", elapsedTime);
	totalBuildingTime = totalBuildingTime + elapsedTime;

	//create nodes and allocate memory for that
	cudaEventRecord(start, 0);
	createParentNodesKernel << <grid1, threads1 >> > (0.0, 0.0, sqrange, sqrange, d_NODE, d_node_counter, d_split_node, maxNodes, d_node_id, d_level, d_positions);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	gpuErrchk(cudaDeviceSynchronize());
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("******** Total Running Time of setting node kernel = %0.5f ms \n", elapsedTime);
	totalBuildingTime = totalBuildingTime + elapsedTime;

	cudaMemcpyFromSymbol(&h_node_allocate, d_node_allocate, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
	printf("number of allocated nodes is: %i \n", h_node_allocate);

	//    err = cudaMemPrefetchAsync(d_NODE, sizeof(NODE)*gpu_nodes , 0);
	//    err = cudaMemPrefetchAsync(d_leave_list, sizeof(int)*leaves  , 0);

	//set the links between childrean and parents
	cudaEventRecord(start, 0);
	finalNodesSetUpKernel << <grid1, threads1 >> > (d_NODE, d_node_counter, d_split_node, maxNodes, d_node_id, d_leave_list, leaf_m_address);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	gpuErrchk(cudaDeviceSynchronize());
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("******** Total Running Time of final node kernel = %0.5f ms \n", elapsedTime);
	totalBuildingTime = totalBuildingTime + elapsedTime;
	cudaMemcpyFromSymbol(&h_point_allocate, d_point_allocate, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
	printf("number of allocated points is: %i number of point is:%i\n", h_point_allocate, numPoints);

	int insert_block_num;
	if (numPoints % INSERT_BLOCK_SIZE == 0) {
		insert_block_num = numPoints / INSERT_BLOCK_SIZE;
	}
	else {
		insert_block_num = numPoints / INSERT_BLOCK_SIZE + 1;
	}
	dim3 grid3(insert_block_num, 1, 1);
	dim3 threads3(INSERT_BLOCK_SIZE, 1, 1);

	//    err = cudaMemPrefetchAsync(d_POINT_x, sizeof(float)*numPoints  , 0);
	//    err = cudaMemPrefetchAsync(d_POINT_y, sizeof(float)*numPoints  , 0);
	//    err = cudaMemPrefetchAsync(d_POINT_id, sizeof(int)*numPoints  , 0);
	//insert into leaf nodes
	cudaEventRecord(start, 0);
	insertIntoLeafNodes << <grid3, threads3 >> > (d_node_id, d_query_POINT_x, d_query_POINT_y, d_query_POINT_id, d_NODE, d_POINT_x, d_POINT_y, d_POINT_id, d_point_node, numPoints, d_POINT_nodeid);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	gpuErrchk(cudaDeviceSynchronize());
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("******** Total Running Time of inserting to leaves = %0.5f ms \n", elapsedTime);
	totalBuildingTime = totalBuildingTime + elapsedTime;

	printf("********** total tree construction time = %0.5f ms \n", totalBuildingTime);


	/////////////////////////////////////////////////
	// sets the CPU historam                      //
	/////////////////////////////////////////////////

	PDH_acnt = numPoints;
	if (argc > 4)
	{
		PDH_res = atof(argv[4]);
	}
	num_buckets = (int)(RANGE * 1.732 / PDH_res) + 1;
	CPU_histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	GPU_histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);




	/* sets CPU and GPU histograms to zero */
	int z = 0;
	for (z = 0; z < num_buckets; z++)
	{
		CPU_histogram[z].d_cnt = 0;
		GPU_histogram[z].d_cnt = 0;
	}

	/*
	for (z = 0; z < numPoints; z++)
	{
	printf("\n(%f,%f) %i", atom_list[z].x_pos, atom_list[z].y_pos, z);
	}
	*/
	gettimeofday(&startTime, &Idunno);
	PDH_baseline();
	report_running_time();

	printf("\nCPU results:\n");
	output_histogram(CPU_histogram);




	/////////////////////////////////////////////////
	// end of the CPU historam                     //
	/////////////////////////////////////////////////



	cudaFree(d_level);
	cudaFree(d_node_counter);
	cudaFree(d_split_node);
	cudaFree(d_positions);
	cudaFree(d_query_POINT_x);
	cudaFree(d_query_POINT_y);
	cudaFree(d_query_POINT_id);
	cudaFree(d_leave_list);
	cudaFree(leaf_m_address);
	cudaFree(d_node_id);
	cudaFree(d_point_node);
	cudaFree(d_POINT_id);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);



	free(h_POINT_x);
	free(h_POINT_y);
	free(h_POINT_id);


	tree_path *tree;
	cudaMalloc((void**)&tree, sizeof(tree_path)*gpu_nodes);

	CUDA_RebuildTree << <1, 1, 1 >> >(d_NODE, gpu_nodes, tree);


	int cuda_block_size = 128;
	if (argc > 5)
	{
		cuda_block_size = atoi(argv[5]);
	}
	int cuda_block_number = ceil(PDH_acnt / cuda_block_size) + 1;

	Quad_Tree_Traversal(num_buckets, cuda_block_number, cuda_block_size, gpu_nodes, tree);

	free(CPU_histogram);
	free(atom_list);

	cudaFree(d_NODE);
	cudaFree(d_POINT_x);
	cudaFree(d_POINT_y);
	cudaFree(d_POINT_nodeid);
	cudaFree(tree);

	return 0;
}