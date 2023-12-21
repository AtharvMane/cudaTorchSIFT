#include<c10/cuda/CUDAException.h>
#include<torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include "keypoints_with_gradients.h"
#include <cuda_runtime.h>

__global__ void doSomethingBranchCUDA(int parentIdx){
    int idx=blockIdx.x*blockDim.x+threadIdx.x+1;
    printf("ParentIdx: %d, BranchIdx: %d", parentIdx, idx);
}
__global__ void doSomethingCUDA(){
    int idx=blockIdx.x*blockDim.x+threadIdx.x+1;
    doSomethingBranchCUDA<<<1,idx>>>(idx);
}

// __device__ dim3 selectorThreads;
// __device__ int diam;
// __device__ int x,y,z;
// __device__ double* img_weights;
__global__ void weightFinder(torch::PackedTensorAccessor64<double,3> gradMagsAccessor, cudaPitchedPtr devPitchedPtr, int pointNum, int diam, int xCenter, int yCenter, int zCenter, int xLim, int yLim){
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    int yImg = yCenter+y-diam/2;
    int xImg = xCenter+x-diam/2;
    if((xImg)<0 || (yImg)<0 || (xImg)>=xLim || (yImg)>=yLim){
        return;
    }
    void* weightPtr = devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch*diam;
    char* plane = (char*) weightPtr + pointNum * slicePitch;
    double* row = (double*) (plane + y*pitch);

    row[x] = gradMagsAccessor[zCenter][xImg][yImg]*expf64(-(powf64(xImg,2)+powf64(yImg,2)));

    
}

__global__ void keyPointsWithGradientsCUDA(cudaPitchedPtr devPitchedPtr, torch::PackedTensorAccessor64<double,2> directionHistogramAccessor, torch::PackedTensorAccessor64<double,1> radiusAccessor, torch::PackedTensorAccessor64<double,2> keypointAccessor, torch::PackedTensorAccessor64<double,3> gradMagsAccessor, torch::PackedTensorAccessor64<int,3> gradDirsAccessor, int xLim, int yLim){
    uint64_t i = threadIdx.x;
    int radius = (int) rintf(radiusAccessor[i]);
    int blocSize = 4;
    dim3 blk(blocSize, blocSize);
    dim3 selectorThreads(radius/2,radius/2);
    int x = (int) rintf(keypointAccessor[i][4]);
    int y = (int) rintf(keypointAccessor[i][3]);
    int z = (int) rintf(keypointAccessor[i][2]);
    __syncthreads();
    weightFinder<<<blk, selectorThreads>>>(gradMagsAccessor, devPitchedPtr, i, 2*radius, x, y, z, xLim, yLim);
    __syncthreads();
    cudaError_t err = cudaGetLastError();

    if(err!=cudaSuccess){
        printf(cudaGetErrorString(err));
        printf("\n");
    }

    void* weightPtr = devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch*2*radius;
    char* plane = (char*) weightPtr + i * slicePitch;
    for(int n=0;n<2*radius;n++){
        double* row = (double*) (plane + n*pitch);
        for(int m=0;m<2*radius;m++){
            directionHistogramAccessor[i][gradDirsAccessor[z][y+n-radius][x+m-radius]] += row[m];
        }
    }

}
