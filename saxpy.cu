#include <stdio.h>

__global__
void singleprecision(int n, float a, float *x, float *y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n){
    y[i] = a * x[i] + y[i];
  }
}

int main(void)
{
  int N = 1<<20;
  float *x, *y, *dev_x, *dev_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&dev_x, N*sizeof(float)); 
  cudaMalloc(&dev_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(dev_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  singleprecision<<<(N+255)/256, 256>>>(N, 2.0f, dev_x, dev_y);

  cudaMemcpy(y, dev_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  cudaFree(dev_x);
  cudaFree(dev_y);
  free(x);
  free(y);
}