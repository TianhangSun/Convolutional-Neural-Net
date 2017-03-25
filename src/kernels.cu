#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>

#define TILE_WIDTH 16

__constant__ int xdims_k[4];
__constant__ int wdims_k[4];
__constant__ int ydims_k[4];
__constant__ int xdims_f[2];
__constant__ int wdims_f[2];

/*__global__ void conv_forward_kernel_basic(float *X, float *W, float *Y){
  int n, m, h, w;
  int W_grid = (ydims_k[2] + TILE_WIDTH - 1) / TILE_WIDTH;
  n = blockIdx.x;
  m = blockIdx.y;
  h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
  w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;
  int C = wdims_k[2]; // in_channel
  int P = wdims_k[0]; // filter_h
  int Q = wdims_k[1]; // filter_w
  
  if(h < ydims_k[1] && w < ydims_k[2]){
    float acc = 0;
    for(int c = 0; c < C; c++){
      for(int p = 0; p < P; p++){
	for(int q = 0; q < Q; q++)
	  acc += X[n * xdims_k[1] * xdims_k[2] * xdims_k[3] + (h + p) * xdims_k[2] * xdims_k[3] + (w + q) * xdims_k[3] + c]
	    * W[p * wdims_k[1] * wdims_k[2] * wdims_k[3] + q * wdims_k[2] * wdims_k[3] + c * wdims_k[3] + m];
      }
    }
    Y[((n * ydims_k[1] + h) * ydims_k[2] + w) * ydims_k[3] + m] = (acc < 0) ? 0 : acc;
  }
}

void conv_forward_host_basic(const float *X, const int xdims[4], const float *W,
                             const int wdims[4], float *Y, const int ydims[4]){
  float *X_device;
  float *W_device;
  float *Y_device;
  int X_size = xdims[0] * xdims[1] * xdims[2] * xdims[3] * sizeof(float);
  int W_size = wdims[0] * wdims[1] * wdims[2] * wdims[3] * sizeof(float);
  int Y_size = ydims[0] * ydims[1] * ydims[2] * ydims[3] * sizeof(float);
  int d_size = 4 * sizeof(int);
  cudaMalloc((void**) &X_device, X_size);
  cudaMalloc((void**) &W_device, W_size);
  cudaMalloc((void**) &Y_device, Y_size);
  cudaMemcpy(X_device, X, X_size, cudaMemcpyHostToDevice);
  cudaMemcpy(W_device, W, W_size, cudaMemcpyHostToDevice);
  cudaMemcpy(Y_device, Y, Y_size, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(xdims_k, xdims, d_size);
  cudaMemcpyToSymbol(wdims_k, wdims, d_size);
  cudaMemcpyToSymbol(ydims_k, ydims, d_size);  
  // std::cout << X_size/sizeof(float) << ", " << W_size/sizeof(float) << ", " << Y_size/sizeof(float) << std::endl;

  int W_grid = (ydims[2] + TILE_WIDTH - 1) / TILE_WIDTH;
  int H_grid = (ydims[1] + TILE_WIDTH - 1) / TILE_WIDTH;
  int Z = H_grid * W_grid;
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 gridDim(ydims[0], ydims[3], Z);
  conv_forward_kernel_basic<<<gridDim, blockDim>>>(X_device, W_device, Y_device);
  cudaDeviceSynchronize();

  cudaMemcpy(Y, Y_device, Y_size, cudaMemcpyDeviceToHost);
  cudaFree(X_device);
  cudaFree(W_device);
  cudaFree(Y_device);
  }*/

/*__global__ void conv_forward_kernel_tiled(half *X, half *W, half *Y){
  int C = xdims_k[3]; // in_channel
  int P = wdims_k[0]; // filter_h
  int Q = wdims_k[1]; // filter_w
  int W_grid = (ydims_k[2] + TILE_WIDTH - 1) / TILE_WIDTH;
  int n, m, h0, w0, h_base, w_base, h, w;
  int X_tile_width = TILE_WIDTH + Q - 1;
  int X_tile_height = TILE_WIDTH + P - 1;
  extern __shared__ half shmem[];
  half *X_shared = &shmem[0];
  half *W_shared = &shmem[X_tile_width * X_tile_height];
  n = blockIdx.x;
  m = blockIdx.y;
  h0 = threadIdx.y;
  w0 = threadIdx.x;
  h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
  w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
  h = h_base + h0;
  w = w_base + w0;

  float acc = 0;
  for(int c = 0; c < C; c++){ // sum over input channels
    if((h0 < P) && (w0 < Q))  // load weight
      W_shared[h0 * Q + w0] = W[h0 * Q * wdims_k[2] * wdims_k[3] + w0 * wdims_k[2] * wdims_k[3] + c * wdims_k[3] + m];
    __syncthreads();
    
    for(int i = h; i < h_base + X_tile_height; i += TILE_WIDTH){ // load tiles
      for(int j = w; j < w_base + X_tile_width; j += TILE_WIDTH)
	X_shared[(i - h_base) * X_tile_width + (j - w_base)] = 
          X[n * xdims_k[1] * xdims_k[2] * xdims_k[3] + i * xdims_k[2] * xdims_k[3] + j * xdims_k[3] + c];
    }
    __syncthreads();

    if(h < ydims_k[1] && w < ydims_k[2]){
      for(int p = 0; p < P; p++){ // sum
	for(int q = 0; q < Q; q++)
	  acc += X_shared[(h0 + p) * X_tile_width + (w0 + q)] * W_shared[p * Q + q];
      }
    }
    __syncthreads();
  }

  if((h < ydims_k[1]) && (w < ydims_k[2]))
    Y[((n * ydims_k[1] + h) * ydims_k[2] + w) * ydims_k[3] + m] = (acc < 0) ? 0 : acc;
}

half* conv_forward_host_tiled(const half *X, const int xdims[4], const half *W, const int wdims[4],
			     half *Y, const int ydims[4], half *in = NULL){
  half *X_device = in;
  half *W_device;
  half *Y_device;
  int X_size = xdims[0] * xdims[1] * xdims[2] * xdims[3] * sizeof(half);
  int W_size = wdims[0] * wdims[1] * wdims[2] * wdims[3] * sizeof(half);
  int Y_size = ydims[0] * ydims[1] * ydims[2] * ydims[3] * sizeof(half);
  int d_size = 4 * sizeof(int);
  if(in == NULL) cudaMalloc((void**) &X_device, X_size);
  cudaMalloc((void**) &W_device, W_size);
  cudaMalloc((void**) &Y_device, Y_size);
  if(in == NULL) cudaMemcpy(X_device, X, X_size, cudaMemcpyHostToDevice);
  cudaMemcpy(W_device, W, W_size, cudaMemcpyHostToDevice);
  //cudaMemcpy(Y_device, Y, Y_size, cudaMemcpyHostToDevice);
  cudaMemset(Y, 0, Y_size);
  cudaMemcpyToSymbol(xdims_k, xdims, d_size);
  cudaMemcpyToSymbol(wdims_k, wdims, d_size);
  cudaMemcpyToSymbol(ydims_k, ydims, d_size);

  int W_grid = (ydims[2] + TILE_WIDTH - 1) / TILE_WIDTH;
  int H_grid = (ydims[1] + TILE_WIDTH - 1) / TILE_WIDTH;
  int Z = H_grid * W_grid;
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 gridDim(ydims[0], ydims[3], Z);
  size_t shmem_size = sizeof(float) * ((TILE_WIDTH + wdims[0] - 1) * (TILE_WIDTH + wdims[1] - 1) + wdims[0] * wdims[1]);
  conv_forward_kernel_tiled<<<gridDim, blockDim, shmem_size>>>(X_device, W_device, Y_device);
  cudaDeviceSynchronize();
  
  //cudaMemcpy(Y, Y_device, Y_size, cudaMemcpyDeviceToHost);
  cudaFree(X_device);
  cudaFree(W_device);
  //cudaFree(Y_device);
  return Y_device;
  }*/

__global__ void average_pool_kernel(float *X, float *Y, int pool_size){
  int n, m, h, w;
  int W_grid = (ydims_k[2] + TILE_WIDTH - 1) / TILE_WIDTH;
  n = blockIdx.x;
  m = blockIdx.y;
  h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
  w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;

  if(h < ydims_k[1] && w < ydims_k[2]){
    float acc = 0;
    for(int p = 0; p < pool_size; p++){
      for(int q = 0; q < pool_size; q++)
        acc += X[n * xdims_k[1] * xdims_k[2] * xdims_k[3] + (pool_size * h + p) * xdims_k[2] * xdims_k[3] +
		 (pool_size * w + q) * xdims_k[3] + m];
    }
    Y[((n * ydims_k[1] + h) * ydims_k[2] + w) * ydims_k[3] + m] = acc / (1.0f * pool_size * pool_size);
  }
}

float* average_pool_host(const float *X, const int xdims[4], const int pool_size,
		       float *Y, const int ydims[4], float *in = NULL){
  float *X_device = in;
  float *Y_device;
  //int X_size = xdims[0] * xdims[1] * xdims[2] * xdims[3] * sizeof(float);
  int Y_size = ydims[0] * ydims[1] * ydims[2] * ydims[3] * sizeof(float);
  int d_size = 4 * sizeof(int);
  //if(in == NULL) cudaMalloc((void**) &X_device, X_size);
  cudaMalloc((void**) &Y_device, Y_size);
  //if(in == NULL) cudaMemcpy(X_device, X, X_size, cudaMemcpyHostToDevice);
  //cudaMemcpy(Y_device, Y, Y_size, cudaMemcpyHostToDevice);
  cudaMemset(Y, 0, Y_size);
  cudaMemcpyToSymbol(xdims_k, xdims, d_size);
  cudaMemcpyToSymbol(ydims_k, ydims, d_size);

  int W_grid = (ydims[2] + TILE_WIDTH - 1) / TILE_WIDTH;
  int H_grid = (ydims[1] + TILE_WIDTH - 1) / TILE_WIDTH;
  int Z = H_grid * W_grid;
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 gridDim(ydims[0], ydims[3], Z);
  average_pool_kernel<<<gridDim, blockDim>>>(X_device, Y_device, pool_size);
  cudaDeviceSynchronize();

  //cudaMemcpy(Y, Y_device, Y_size, cudaMemcpyDeviceToHost);
  cudaFree(X_device);
  //cudaFree(Y_device);
  return Y_device;
}

/*__global__ void fully_forward_kernel(float *X, float *W, float *Y, bool relu){
  int i = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int j = blockIdx.y * TILE_WIDTH + threadIdx.y;
  if(i < xdims_f[0] && j < wdims_f[1]){
    float sum = 0;
    for (int k = 0; k < xdims_f[1]; k++) {
      sum += X[i * xdims_f[1] + k] * W[k * wdims_f[1] + j];
    }
    Y[i * wdims_f[1] + j] = (relu && (sum < 0)) ? 0: sum;
  }
}

float* fully_forward_host(const float *X, const int xdims[2], float *W, const int wdims[2],
			  float *Y, const int ydims[2], float *in = NULL, bool copy = false){
  float *X_device = in;
  float *W_device;
  float *Y_device;
  //int X_size = xdims[0] * xdims[1] * sizeof(float);
  int W_size = wdims[0] * wdims[1] * sizeof(float);
  int Y_size = ydims[0] * ydims[1] * sizeof(float);
  int d_size = 2 * sizeof(int);
  //if(in == NULL) cudaMalloc((void**) &X_device, X_size);
  cudaMalloc((void**) &W_device, W_size);
  cudaMalloc((void**) &Y_device, Y_size);
  //if(in == NULL) cudaMemcpy(X_device, X, X_size, cudaMemcpyHostToDevice);
  cudaMemcpy(W_device, W, W_size, cudaMemcpyHostToDevice);
  //cudaMemcpy(Y_device, Y, Y_size, cudaMemcpyHostToDevice);
  cudaMemset(Y, 0, Y_size);
  cudaMemcpyToSymbol(xdims_f, xdims, d_size);
  cudaMemcpyToSymbol(wdims_f, wdims, d_size);
  
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 gridDim((xdims[0]+TILE_WIDTH-1)/TILE_WIDTH, (wdims[1]+TILE_WIDTH-1)/TILE_WIDTH, 1);
  fully_forward_kernel<<<gridDim, blockDim>>>(X_device, W_device, Y_device, !copy);
  cudaDeviceSynchronize();

  if(copy) cudaMemcpy(Y, Y_device, Y_size, cudaMemcpyDeviceToHost);
  cudaFree(X_device);
  cudaFree(W_device);
  //cudaFree(Y_device);
  return Y_device;
  }*/

__global__ void fully_forward_kernel_tiled(float *A, float *B, float *C, bool relu) {
  extern __shared__ float shmemmrelu[];
  float *Ads = &shmemmrelu[0];
  float *Bds = &shmemmrelu[TILE_WIDTH * TILE_WIDTH];

  int bx=blockIdx.x;int by=blockIdx.y;
  int tx=threadIdx.x;int ty=threadIdx.y;
  int Row=by*TILE_WIDTH+ty;
  int Col=bx*TILE_WIDTH+tx;
  float Cvalue=0;
  for (int ph = 0; ph < (xdims_f[1] + TILE_WIDTH - 1) / TILE_WIDTH; ++ph){
    if ((Row < xdims_f[0]) && (ph * TILE_WIDTH + tx < xdims_f[1]))
      Ads[ty * TILE_WIDTH + tx] = A[Row * xdims_f[1] + ph * TILE_WIDTH + tx];
    else
      Ads[ty * TILE_WIDTH + tx] = 0.0;
    if((ph * TILE_WIDTH + ty < wdims_f[0]) && (Col < wdims_f[1]))
      Bds[ty * TILE_WIDTH + tx] = B[((ph * TILE_WIDTH) + ty) * wdims_f[1] + Col];
    else
      Bds[ty * TILE_WIDTH + tx] = 0.0;
    __syncthreads();
    
    for(int k = 0; k < TILE_WIDTH; ++k)
      Cvalue += Ads[ty * TILE_WIDTH + k] * Bds[k * TILE_WIDTH + tx];
    __syncthreads();
  }

  if ((Row < xdims_f[0]) && (Col < wdims_f[1]))
    C[Row * wdims_f[1] + Col] = (Cvalue < 0 && relu) ? 0 : Cvalue;
}

float* fully_forward_host_tiled(const float *X, const int xdims[2], float *W, const int wdims[2],
		     float *Y, const int ydims[2], float *in, bool copy = false){
  float *X_device = in;
  float *W_device;
  float *Y_device;
  //int X_size = xdims[0] * xdims[1] * sizeof(float);
  int W_size = wdims[0] * wdims[1] * sizeof(float);
  int Y_size = ydims[0] * ydims[1] * sizeof(float);
  int d_size = 2 * sizeof(int);
  //cudaMalloc((void**) &X_device, X_size);
  cudaMalloc((void**) &W_device, W_size);
  cudaMalloc((void**) &Y_device, Y_size);
  //cudaMemcpy(X_device, X, X_size, cudaMemcpyHostToDevice);
  cudaMemcpy(W_device, W, W_size, cudaMemcpyHostToDevice);
  cudaMemcpy(Y_device, Y, Y_size, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(xdims_f, xdims, d_size);
  cudaMemcpyToSymbol(wdims_f, wdims, d_size);
  
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 gridDim((wdims[1]+TILE_WIDTH-1)/TILE_WIDTH,(xdims[0]+TILE_WIDTH-1)/TILE_WIDTH, 1);
  size_t shmemm_size = sizeof(float) * (TILE_WIDTH * TILE_WIDTH * 2);
  fully_forward_kernel_tiled<<<gridDim, blockDim, shmemm_size>>>(X_device, W_device, Y_device, !copy);
  cudaDeviceSynchronize();

  if(copy) cudaMemcpy(Y, Y_device, Y_size, cudaMemcpyDeviceToHost);
  cudaFree(X_device);
  cudaFree(W_device);
  //cudaFree(Y_device);
  return Y_device;
}

__global__ void gemmrelu_conv_kernel_merge(float *A, float *B, float *C) {
  extern __shared__ float shmemmrelu[];
  
  float *Ads = &shmemmrelu[0];
  float *Bds = &shmemmrelu[TILE_WIDTH * TILE_WIDTH];
  // Y[n, output height , output width, m] = 0  // W[p filter_h, q filter_w, c,  m]   // X[n, h + p,w + q,c]
  int n = blockIdx.z;
  int numARows = ydims_k[3];
  int numAColumns = xdims_k[3] * wdims_k[0] * wdims_k[1];
  int numBRows = xdims_k[3] * wdims_k[0] * wdims_k[1];
  int numBColumns = ydims_k[1] * ydims_k[2];
  int numCRows = ydims_k[3];
  int numCColumns = ydims_k[1] * ydims_k[2];
  int bx=blockIdx.x; int by=blockIdx.y;
  int tx=threadIdx.x; int ty=threadIdx.y;
  int Row=by*TILE_WIDTH+ty;
  int Col=bx*TILE_WIDTH+tx;
  float Cvalue=0;
  for (int ph=0;ph<(numAColumns+TILE_WIDTH-1)/TILE_WIDTH;++ph){

    if ((Row<numARows)&&(ph*TILE_WIDTH+tx<numAColumns)){
      int m = by * TILE_WIDTH + ty;
      int c = (ph * TILE_WIDTH + tx)/ (wdims_k[0] * wdims_k[1]);
      int p = ((ph * TILE_WIDTH + tx) % (wdims_k[0] * wdims_k[1])) / wdims_k[1];
      int q = ((ph * TILE_WIDTH + tx) % (wdims_k[0] * wdims_k[1])) % wdims_k[1];
      Ads[ty * TILE_WIDTH + tx]=A[p * wdims_k[1] * wdims_k[2] * wdims_k[3] + q * wdims_k[2] * wdims_k[3] + c * wdims_k[3] + m];
    }
    else
      Ads[ty * TILE_WIDTH + tx]=0.0;
    if((ph * TILE_WIDTH + ty<numBRows)&&(Col<numBColumns)){
      int cx = (ph * TILE_WIDTH + ty) / (wdims_k[0] * wdims_k[1]);
      int px = ((ph * TILE_WIDTH + ty) % (wdims_k[0] * wdims_k[1])) / wdims_k[1];
      int qx = ((ph * TILE_WIDTH + ty) % (wdims_k[0] * wdims_k[1])) % wdims_k[1];
      int h_out = (bx * TILE_WIDTH + tx) / ydims_k[2];
      int w_out = (bx * TILE_WIDTH + tx) % ydims_k[2];
     Bds[ty * TILE_WIDTH + tx] = 
       B[n * xdims_k[1] * xdims_k[2] * xdims_k[3] + (h_out + px) * xdims_k[2] * xdims_k[3] +(w_out + qx) * xdims_k[3] + cx];}
    else
      Bds[ty * TILE_WIDTH + tx] = 0.0;
    __syncthreads();
    
    for(int k=0; k<TILE_WIDTH; ++k){
      Cvalue += Ads[ty * TILE_WIDTH + k]*Bds[k * TILE_WIDTH + tx];}
    __syncthreads();
  }

  if ((Row<numCRows)&&(Col<numCColumns)){
    atomicAdd(&C[n * ydims_k[1] * ydims_k[2] * ydims_k[3] + (Col / ydims_k[2]) * ydims_k[2] * ydims_k[3]
    	   + (Col % ydims_k[2]) * ydims_k[3] + Row], (Cvalue < 0) ? 0 : Cvalue);
    //C[n * ydims_k[1] * ydims_k[2] * ydims_k[3] + (Col / ydims_k[2]) * ydims_k[2] * ydims_k[3]
    //+ (Col % ydims_k[2]) * ydims_k[3] + Row] = (Cvalue < 0) ? 0 : Cvalue;
  }
}


float* convLayer_forward_merge(float *X, float *W, float *Y, const int xdims[4], const int ydims[4],
			       const int wdims[4], float *in = NULL){
  int d_size = sizeof(int) * 4;
  int W_size = sizeof(float) * wdims[0] * wdims[1] *wdims[2] * wdims[3];
  int Y_size = ydims[0] * ydims[1] * ydims[2] * ydims[3] * sizeof(float);
  int X_size = xdims[0] * xdims[1] * xdims[2] * xdims[3] * sizeof(float);
  size_t shmemm_size = sizeof(float) * (TILE_WIDTH * TILE_WIDTH * 2);

  float *W_device;          //// Y[n, output height , output width, m] = 0 // W[p filter_h, q filter_w, c,  m]   X[n, h + p,w + q,c]
  float *X_device = in;
  float *Y_device;

  cudaMalloc((void**) &Y_device, Y_size);
  cudaMalloc((void**) &W_device, W_size);
  if(in == NULL) cudaMalloc((void**) &X_device, X_size);
  cudaMemset(Y_device, 0, Y_size);

  cudaMemcpy(W_device, W, W_size, cudaMemcpyHostToDevice);
  if(in == NULL) cudaMemcpy(X_device, X, X_size, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(wdims_k, wdims, d_size);
  cudaMemcpyToSymbol(xdims_k, xdims, d_size);
  cudaMemcpyToSymbol(ydims_k, ydims, d_size);

  dim3 blockDim1(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 gridDim3((ydims[1] * ydims[2] + TILE_WIDTH-1)/TILE_WIDTH, (ydims[3]+TILE_WIDTH-1)/TILE_WIDTH, ydims[0]);
  
  gemmrelu_conv_kernel_merge<<< gridDim3, blockDim1, shmemm_size >>>(W_device, X_device, Y_device);
  cudaDeviceSynchronize();

  //cudaMemcpy(Y, Y_device, Y_size, cudaMemcpyDeviceToHost);

  cudaFree(X_device);
  //cudaFree(Y_device);
  cudaFree(W_device);
  return Y_device;
}

__global__ void float_to_half_kernel(float *in, half *out, int size){
  int pos = blockIdx.x * TILE_WIDTH * TILE_WIDTH + threadIdx.x;
  if(pos < size) out[pos] = __float2half(in[pos]);
}

half* float_to_half_host(float *in, int size){
  float *in_device;
  half *out_device;
  cudaMalloc((void**) &in_device, sizeof(float) * size);
  cudaMalloc((void**) &out_device, size * sizeof(half));
  cudaMemcpy((void**) &in_device, in, sizeof(float) * size, cudaMemcpyHostToDevice);
  cudaMemset(out_device, 0, size * sizeof(half));

  dim3 blockDim(TILE_WIDTH * TILE_WIDTH, 1, 1);
  dim3 gridDim((size + TILE_WIDTH * TILE_WIDTH - 1) / (TILE_WIDTH * TILE_WIDTH), 1, 1);
  float_to_half_kernel<<<gridDim, blockDim>>>(in_device, out_device, size);
  cudaDeviceSynchronize();
  cudaFree(in_device);
  return out_device;
}

__global__ void half_to_float_kernel(half *in, float *out, int size){
  int pos = blockIdx.x * TILE_WIDTH * TILE_WIDTH + threadIdx.x;
  if(pos < size) out[pos] = __half2float(in[pos]);
}

float* half_to_float_host(half *in, int size){
  half *in_device;
  float *out_device;
  cudaMalloc((void**) &in_device, sizeof(half) * size);
  cudaMalloc((void**) &out_device, sizeof(float) * size);
  cudaMemcpy((void**) &in_device, in, sizeof(half) * size, cudaMemcpyHostToDevice);
  cudaMemset(out_device, 0, size * sizeof(half));

  dim3 blockDim(TILE_WIDTH * TILE_WIDTH, 1, 1);
  dim3 gridDim((size + TILE_WIDTH * TILE_WIDTH - 1) / (TILE_WIDTH * TILE_WIDTH), 1, 1);
  half_to_float_kernel<<<gridDim, blockDim>>>(in_device, out_device, size);
  cudaDeviceSynchronize();
  cudaFree(in_device);
  return out_device;
}
