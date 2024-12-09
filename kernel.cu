#include <stdio.h>

#define TILE_SIZE 0

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    __shared__ float A_shared[TILE_SIZE][TILE_SIZE]; 
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float Pvalue = 0.0f;

   int i = 0;
   while (i < (k - 1) / TILE_SIZE + 1) {
    if (row < m && i * TILE_SIZE + threadIdx.x < k)
        A_shared[threadIdx.y][threadIdx.x] = A[row * k + i * TILE_SIZE + threadIdx.x];
    else
        A_shared[threadIdx.y][threadIdx.x] = 0.0;

    if (i * TILE_SIZE + threadIdx.y < k && col < n)
        B_shared[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * n + col];
    else
        B_shared[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    if (row < m && col < n) {
        int j = 0;
        while (j < TILE_SIZE) {
            Pvalue += A_shared[threadIdx.y][j] * B_shared[j][threadIdx.x];
            j++;
        }
    }

    __syncthreads();
    i++;

    }

    if (row < m && col < n) {
        C[row * n + col] = Pvalue;
    }
    /*************************************************************************/
}

void basicSgemm(int m, int n, int k, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;
	
    /*************************************************************************/
    //INSERT CODE HERE
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n-1) / BLOCK_SIZE + 1, ((m-1)/BLOCK_SIZE+1));
    /*************************************************************************/

    // Invoke CUDA kernel -----------------------------------------------------
    mysgemm<<<dimGrid, dimBlock>>>(m,n,k,A,B,C);
    /*************************************************************************/
    //INSERT CODE HERE

	cudaDeviceSynchronize();
    /*************************************************************************/
}


