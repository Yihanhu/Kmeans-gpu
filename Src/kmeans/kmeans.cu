#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include "MNIST.h"
#include <vector>
#include <algorithm>

using namespace std;
using namespace chrono;
#define K 10
#define DIMENSION 784
#define N 10000
#define IterMax 5000


__global__ void update(float* d_mean, float* d_data, int* d_label) {

	__shared__ float s_mean[10][DIMENSION];
	unsigned int dataid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int tid = threadIdx.x % (N/10);
	int clabel = d_label[dataid];

	for (int i = 0; i < DIMENSION; i++) {
		atomicAdd(&d_mean[clabel*DIMENSION + i],d_data[dataid + i*N]);
	}
	
	__syncthreads();
}


__global__ void divide(float* d_mean, int* count) {
	unsigned int clabel = blockIdx.x;
	unsigned int tid = threadIdx.x;
	d_mean[clabel*DIMENSION + tid] = d_mean[clabel*DIMENSION + tid] / ((double)count[clabel] );
}



__global__ void label_init(int *d_label,int* d_count, curandState *states) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(clock() + tid, tid, 0, &states[tid]);
	d_label[tid] = (int) (curand_uniform(&states[tid]) * K);
	atomicAdd(&d_count[d_label[tid]],1);

}


__global__ void assign_labels (float* d_data, float* d_means, int* d_label, int* d_count) {


	__shared__ float s_data[DIMENSION];
	__shared__ float s_dist[1024];
	s_data[threadIdx.x] = d_data[blockIdx.x * DIMENSION + threadIdx.x];
	if (threadIdx.x < (1024 - DIMENSION)) {
		s_dist[DIMENSION + threadIdx.x] = 0;
	}
	__syncthreads();


	float dist_min = FLT_MAX;
	int t_label = 10;

	for (int i = 0; i < K; i++) {

		s_dist[threadIdx.x] = (s_data[threadIdx.x] - d_means[i * DIMENSION + threadIdx.x]) * (s_data[threadIdx.x] - d_means[i * DIMENSION + threadIdx.x]);
		__syncthreads();

		for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
			if (threadIdx.x < stride) {
				s_dist[threadIdx.x] += s_dist[threadIdx.x + stride];
			}
			__syncthreads();
		}

		if (s_dist[0] < dist_min) {
			dist_min = s_dist[0];
			t_label = i;
		}

	}

	if(threadIdx.x == 0){
		d_label[blockIdx.x] = t_label;
		atomicAdd(&d_count[t_label],1);
	}
	
}


int main(int argc, char **argv) {


	// load datase
	MNIST mnist = MNIST("./");
	printf("Data Dimension: %d\n",mnist.testData[0].pixelData.size());

	float **test_image = new float*[N];
	test_image[0] = new float[N * DIMENSION];
	for (int i = 1; i < N; ++i) test_image[i] = test_image[i - 1] + DIMENSION;

	float **test_image_T = new float*[DIMENSION];
	test_image_T[0] = new float[N * DIMENSION];
	for (int i = 1; i < DIMENSION; ++i) test_image_T[i] = test_image_T[i - 1] + N;
	
	int gt_label[N];
	for (int i = 0; i <  N; i++) {
		gt_label[i] = (int)mnist.testData[i].label;
		for (int j = 0; j < DIMENSION; j++) {
			test_image[i][j] = (float)mnist.testData[i].pixelData[j];
			test_image_T[j][i] = test_image[i][j];
		}
	}

	// cudaMalloc
	float *d_data,*d_data_T, *d_means;
	int *d_label, *d_count;
	curandState *states;
	cudaMalloc((float **)&d_data, N * DIMENSION * sizeof(float));
	cudaMalloc((float **)&d_data_T, N * DIMENSION * sizeof(float));
	cudaMalloc((float **)&d_means, K * DIMENSION * sizeof(float));
	cudaMalloc((int **)&d_label, N * sizeof(int));
	cudaMalloc((int **)&d_count, K * sizeof(int));
	cudaMalloc((void **)&states, sizeof(curandState) * N);

	// Memcpy to global 
	cudaMemcpy(d_data, test_image[0], N * DIMENSION * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_data_T, test_image_T[0], N * DIMENSION * sizeof(float), cudaMemcpyHostToDevice);
	// Initialization: randomly assign labels
	dim3 grid(10, 1, 1);
	dim3 block(N/10, 1, 1);
	cudaMemset(d_count, 0, K * sizeof(int));
	label_init <<<grid, block>>> (d_label, d_count, states);

	//start timer
	clock_t begin = clock();

	// main loop
	dim3 grid2(N, 1, 1);
	dim3 block2(DIMENSION, 1, 1);

	for (int iter = 0; iter < IterMax; iter++) {

		// update means
		cudaMemset(d_means, 0.0, DIMENSION * K * sizeof(float));	
		update <<<grid, block>>> (d_means,d_data_T, d_label);
		divide <<<K, DIMENSION>>> (d_means, d_count);

		// assign labels
		cudaMemset(d_count, 0, K * sizeof(int));
		assign_labels <<<grid2, block2>>> (d_data, d_means, d_label, d_count);

	}
	
	clock_t end = clock();
	double elapsedTime = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Time spent: %fs\n", elapsedTime);


	// Mem copyback
	int* predict_label =  (int*)malloc(N*sizeof(int));
	cudaMemcpy(predict_label, d_label, N * sizeof(int), cudaMemcpyDeviceToHost);
	
	FILE *fptr = fopen("prediction.txt", "w");
	for (int i = 0; i < N; i++) {
		fprintf(fptr, "%d\n", predict_label[i]);
	}
	fclose(fptr);
	
	// make prediction
	vector<vector<int>> clusters(10, vector<int>(11,0));
	for (int i = 0; i < N; i++) {
		clusters[gt_label[i]][predict_label[i]] ++;
		clusters[gt_label[i]][10] ++;
	}
	float acc[10];
	for (int i = 0; i < 10; i++) {
		int max_ele = INT_MIN;
		for (int j = 0; j < 10; j++) {
			if (max_ele < clusters[gt_label[i]][j]) {
				max_ele = clusters[gt_label[i]][j];
			}
		}
		acc[i] = (float) max_ele/ (float)clusters[gt_label[i]][10];
	}

	// Print results
	float max_ele = FLT_MIN;
	for (int j = 0; j < 10; j++) {
		if (max_ele < acc[j]) {
			max_ele = acc[j];
		}
	}
	printf("Max Accuracy: %f%\n", max_ele*100);
	

	//free
	cudaFree(states);
	cudaFree(d_count);
	cudaFree(d_label);
	cudaFree(d_means);
	cudaFree(d_data);
	cudaFree(d_data_T);

	free(test_image);
	free(test_image_T);
	free(predict_label);

	getchar();
	return 0;

}

