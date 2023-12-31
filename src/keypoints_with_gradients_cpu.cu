#include<c10/cuda/CUDAException.h>
#include<torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include "keypoints_with_gradients.cuh"
#include <cuda_runtime.h>

using namespace torch::indexing;

torch::Tensor keypointsWithGradientsAndDescriptors(at::Tensor keypoints, at::Tensor gaussGradMags, at::Tensor gaussGradDirs, int granularity, double initialSigma, double scaleFactor, int numIntervals){
    assert(keypoints.is_contiguous() && keypoints.is_cuda());
    assert(gaussGradMags.is_contiguous() && gaussGradMags.is_cuda());
    assert(gaussGradDirs.is_contiguous() && gaussGradDirs.is_cuda());

    torch::Tensor directionHistogram =torch::zeros({keypoints.size(0), (int)(360/granularity)+1}, torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));
    gaussGradDirs = (gaussGradDirs/granularity).to(torch::kInt);

    //coutTensorShape(directionHistogram.sizes(), "directionHistogram");
    torch::Tensor scales = scaleFactor * initialSigma * torch::exp2(keypoints.index({Slice(),2})/numIntervals);
    torch::Tensor radii = 4*scales;
    // coutTensorShape(keypoints.sizes(), "keypoints");

    int maxRadius = torch::round(torch::max(radii)).to(torch::kLong).item().toInt();
    
    
    cudaExtent extent = make_cudaExtent(2*maxRadius* sizeof(float),2*maxRadius, keypoints.size(0));
    cudaPitchedPtr devPitchedPtr;
    cudaMalloc3D(&devPitchedPtr, extent);

    torch::PackedTensorAccessor64<double,2> directionHistogramAccessor = directionHistogram.packed_accessor64<double,2>();
    torch::PackedTensorAccessor64<double,2> keypointAccessor = keypoints.packed_accessor64<double,2>();
    torch::PackedTensorAccessor64<double, 1> radiusAccessor = radii.packed_accessor64<double,1>();
    torch::PackedTensorAccessor64<double,3> gradMagsAccessor = gaussGradMags.packed_accessor64<double,3>();
    torch::PackedTensorAccessor64<int,3> gradDirsAccessor = gaussGradDirs.packed_accessor64<int,3>(); 
    cudaDeviceSynchronize();
    createWeightTiles<<<2, (keypoints.size(0)+1)/2>>>(keypoints.size(0), devPitchedPtr, directionHistogramAccessor, radiusAccessor, keypointAccessor, gradMagsAccessor, gradDirsAccessor, gaussGradMags.size(2),gaussGradMags.size(1), keypoints.size(0));
    cudaError_t err = cudaGetLastError();
    if(err!=cudaSuccess){
        printf("%s\n",cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    createHistogram<<<2, (keypoints.size(0)+1)/2>>>(devPitchedPtr, directionHistogramAccessor, radiusAccessor, keypointAccessor, gradMagsAccessor, gradDirsAccessor, keypoints.size(0));
    err = cudaGetLastError();
    if(err!=cudaSuccess){
        printf("%s\n",cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    cudaFree(devPitchedPtr.ptr);
    return directionHistogram;
}
