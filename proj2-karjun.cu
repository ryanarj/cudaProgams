
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "device_launch_parameters.h"
#include <curand_kernel.h>

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
int threads;			/* total number of threads in the histogram */
double PDH_res;			/* value of w                             */
atom *atom_list;		/* list of all data points                */

struct timezone Idunno;	
struct timeval startTime, endTime;

double p2p_distanceOriginal(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


__device__ double
p2p_distance(atom *atom, int ind1, int ind2) {
	
	double x1 = atom[ind1].x_pos;
	double x2 = atom[ind2].x_pos;
	double y1 = atom[ind1].y_pos;
	double y2 = atom[ind2].y_pos;
	double z1 = atom[ind1].z_pos;
	double z2 = atom[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}

double report_running_time_gpu() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for GPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}

int PDH_baselineOrginal() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distanceOriginal(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}

__global__ void
PDH_baseline(bucket *histogram, atom *atom, double weight) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	printf("i: %d\n",i);
    printf("j: %d\n",j);
	// Get the distance and then the position of the histogram with 1
	if (i < j) {
		double distance = p2p_distance(atom, i, j);
		int position = (int) (distance / weight);
		histogram[position].d_cnt++;
		// Using Barrier Synchronization
		__syncthreads();
	}
}

__global__ void
generate_data(atom *a, long long a_num){
	int i = threadIdx.x + (blockIdx.x * blockDim.x);

	// Create random numbers, using clock so no two values can ever be the same
	curandState state;
	curand_init(((unsigned long long)clock() + i) * BOX_SIZE, RAND_MAX, 1, &state);
	a[i].x_pos = curand_uniform_double(&state) * BOX_SIZE;
	a[i].y_pos = curand_uniform_double(&state) * BOX_SIZE;
	a[i].z_pos = curand_uniform_double(&state) * BOX_SIZE;
	
	// Using Barrier Synchronization
	__syncthreads();
}

long long output_histogram(){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
	return total_cnt;
}


int main(int argc, char const *argv[])
{
	if (argc > 2){
		PDH_acnt = atoi(argv[1]);	// Number of atoms
		PDH_res = atof(argv[2]);	// Input Distance: W
		threads = atoi(argv[3]);    // Number of threads
	} else {
		printf("Invalid amount of arguments!!\n Needs a number of atoms, the distance amount and number of threads.");
		return 0;
	}
	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	size_t histogramSize = sizeof(bucket)*num_buckets;
	size_t atomSize = sizeof(atom)*PDH_acnt;
	histogram = (bucket *)malloc(histogramSize);
	atom_list = (atom *)malloc(atomSize);
	bucket *d_histogram = NULL;
	atom *d_atom_list = NULL;
	atom *d_atom_list2 = NULL;
	double difference_time1, difference_time2; 
	long long  difference_t1, difference_t2;

	srand(1);
	/* generate data following a uniform distribution */
	cudaMalloc((void**) &d_atom_list2, atomSize);
	cudaMemcpy(d_atom_list2, atom_list, atomSize, cudaMemcpyHostToDevice);
	generate_data <<<ceil(PDH_acnt/threads), threads>>> (d_atom_list2, PDH_acnt);
	cudaMemcpy(atom_list, d_atom_list2, atomSize, cudaMemcpyDeviceToHost);
	cudaFree(d_atom_list2);

	/* start counting time */
	gettimeofday(&startTime, &Idunno);
	
	/* call CPU single thread version to compute the histogram */
	PDH_baselineOrginal();
	
	/* check the total running time */ 
	difference_time1 = report_running_time();
	
	/* print out the histogram */
	difference_t1 = output_histogram();

	// Allocate memory to GPU arrays and then copy the data from the CPU arrays
	cudaMalloc((void**) &d_histogram, histogramSize);
	cudaMalloc((void**) &d_atom_list, atomSize);
	cudaMemcpy(d_histogram, histogram, histogramSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_atom_list, atom_list, atomSize, cudaMemcpyHostToDevice);

	/* start counting time */
	gettimeofday(&startTime, &Idunno);

	// Add the cuda kernal start time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Launch the kernal and perform calcualtions with the GPU PDH_baseline
	PDH_baseline <<<ceil(PDH_acnt/threads), threads>>> (d_histogram, d_atom_list, PDH_res);

	// Get the end time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to generate: %0.5f ms \n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(histogram, d_histogram, histogramSize, cudaMemcpyDeviceToHost);

	/* check the total running time */ 
	difference_time2 = report_running_time_gpu();

	// Print the histogram
	difference_t2 = output_histogram();

	printf("\nDifference Of the two histograms.\n");
	printf("Time to generate on the Kernal with GPU: %0.5f ms \n", elapsedTime);
	printf("Running time for CPU version: %lf\n", difference_time1);
	printf("Running time for GPU version: %lf\n", difference_time2);
	printf("Total distance for CPU version: %lld\n", difference_t1);
	printf("Total distance for GPU version: %lld\n", difference_t2);

	// Free the GPU(device) and the CPU(host) arrays
	cudaFree(d_histogram);
	cudaFree(d_atom_list);
	free(histogram);
	free(atom_list);

	return 0;
}
