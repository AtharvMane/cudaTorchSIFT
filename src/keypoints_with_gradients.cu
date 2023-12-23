#include<c10/cuda/CUDAException.h>
#include<torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include "keypoints_with_gradients.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
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
__global__ void createSingleWeightTile(torch::PackedTensorAccessor64<double,3> gradMagsAccessor, cudaPitchedPtr devPitchedPtr, long pointNum, int diam, int xCenter, int yCenter, int zCenter, int xLim, int yLim, double sigmaInv){
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    int yImg = yCenter+y-diam/2;
    int xImg = xCenter+x-diam/2;


    if((xImg)>=0 && (yImg)>=0 && (xImg)<xLim && (yImg)<yLim){
        void* weightPtr = devPitchedPtr.ptr;
        size_t pitch = devPitchedPtr.pitch;
        size_t slicePitch = pitch*diam;
        char* plane = (char*) weightPtr + pointNum * slicePitch;
        double* row = (double*) (plane + y*pitch);
        row[x] = gradMagsAccessor[zCenter][yImg][xImg]*exp(-(pow(x-diam/2,2)+pow(y-diam/2,2))*sigmaInv);
    }
}

__global__ void createWeightTiles(int size, cudaPitchedPtr devPitchedPtr, torch::PackedTensorAccessor64<double,2> directionHistogramAccessor, torch::PackedTensorAccessor64<double,1> radiusAccessor, torch::PackedTensorAccessor64<double,2> keypointAccessor, torch::PackedTensorAccessor64<double,3> gradMagsAccessor, torch::PackedTensorAccessor64<int,3> gradDirsAccessor, int xLim, int yLim){
    cg::grid_group grp = cg::this_grid();
    
    uint64_t i = threadIdx.x;
    double sigmaInv = 2*pow(2/radiusAccessor[0],2);
    int radius = (int) rintf(radiusAccessor[i]);
    int blocSize = 4;
    dim3 blk(blocSize, blocSize);
    dim3 selectorThreads(radius/2,radius/2);
    int x = (int) rint(keypointAccessor[i][4]);
    int y = (int) rint(keypointAccessor[i][3]);
    int z = (int) rint(keypointAccessor[i][2]);
    createSingleWeightTile<<<blk, selectorThreads>>>(gradMagsAccessor, devPitchedPtr, i, 2*radius, x, y, z, xLim, yLim, sigmaInv);
    cudaError_t err = cudaGetLastError();
    if(err!=cudaSuccess){
        printf("%s\n",cudaGetErrorString(err));
    }
}

__global__ void createHistogram(cudaPitchedPtr devPitchedPtr, torch::PackedTensorAccessor64<double,2> directionHistogramAccessor, torch::PackedTensorAccessor64<double,1> radiusAccessor, torch::PackedTensorAccessor64<double,2> keypointAccessor, torch::PackedTensorAccessor64<double,3> gradMagsAccessor, torch::PackedTensorAccessor64<int,3> gradDirsAccessor){
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    int radius = radiusAccessor[idx];
    void* weightPtr = devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch*2*radius;
    char* plane = (char*) weightPtr + idx * slicePitch;
    int x = (int) rint(keypointAccessor[idx][4]);
    int y = (int) rint(keypointAccessor[idx][3]);
    int z = (int) rint(keypointAccessor[idx][2]);

    for(int n=0;n<2*radius;n++){
        double* row = (double*) (plane + n*pitch);
        for(int m=0;m<2*radius;m++){
            directionHistogramAccessor[idx][gradDirsAccessor[z][y+n-radius][x+m-radius]] += row[m];
        }
    }
}
