
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;

bucket *histogram;		/* list of all buckets in the histogram   */
long long PDH_acnt;		/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double PDH_res;			/* value of w                             */
atom *atom_list;		/* list of all data points                */

__device__ double
p2p_distance(atom *a, int ind1, int ind2) {
	double x1 = a[ind1].x_pos;
	double x2 = a[ind2].x_pos;

	double y1 = a[ind1].y_pos;
	double y2 = a[ind2].y_pos;

	double z1 = a[ind1].z_pos;
	double z2 = a[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

__global__ void 
PDH_baseline(bucket *h, atom *atom, double w, int size) {
	int i, j, pos;
	double dist;
	// Apply block index to randomize the threads
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = i + 1;

	for (int a = j; a < size; a++) {
		dist = p2p_distance(atom, i, a);
		pos = (int) (dist / w);
		atomicAdd( &h[pos].d_cnt, 1);
	}
}

void output_histogram(bucket *h){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", h[i].d_cnt);
		total_cnt += h[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}


int main(int argc, char const *argv[])
{
	PDH_acnt = atoi(argv[1]);	// Number of atoms
	PDH_res = atof(argv[2]);	// Input Distance: W
	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	size_t histogramSize = sizeof(bucket)*num_buckets;
	size_t atomSize = sizeof(atom)*PDH_acnt;
	histogram = (bucket *)malloc(histogramSize);
	atom_list = (atom *)malloc(atomSize);
	bucket *d_histogram = NULL;
	atom *d_atom_list = NULL;

	srand(1);
	/* generate data following a uniform distribution */
	for(int i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}

	// Allocate memory to GPU arrays and then copy the data from the CPU arrays
	cudaMalloc((void**) &d_histogram, histogramSize);
	cudaMalloc((void**) &d_atom_list, atomSize);
	cudaMemcpy(d_histogram, histogram, histogramSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_atom_list, atom_list, atomSize, cudaMemcpyHostToDevice);

	// Launch the kernal and perform calcualtions with the GPU PDH_baseline
	PDH_baseline <<<ceil(PDH_acnt/64), 64>>> (d_histogram, d_atom_list, PDH_res, PDH_acnt);
	cudaMemcpy(histogram, d_histogram, histogramSize, cudaMemcpyDeviceToHost);

	// Print the histogram
	output_histogram(histogram);

	// Free the GPU(device) and the CPU(host) arrays
	cudaFree(d_histogram);
	cudaFree(d_atom_list);
	free(histogram);
	free(atom_list);
	cudaDeviceReset();

	return 0;
}


