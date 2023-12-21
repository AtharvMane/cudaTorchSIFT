#include<c10/cuda/CUDAException.h>
#include<torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include "keypoints_with_gradients.h"
#include <cuda_runtime.h>

// void do_something(){
//     std::cout<<"doing_something"<<std::endl;
//     doSomethingCUDA<<<1,3>>>();
// }

using namespace torch::indexing;

torch::Tensor keypointsWithGradients(at::Tensor keypoints, at::Tensor gaussGradMags, at::Tensor gaussGradDirs, int granularity, double initialSigma, double scaleFactor, int numIntervals){
    assert(keypoints.is_contiguous() && keypoints.is_cuda());
    assert(gaussGradMags.is_contiguous() && gaussGradMags.is_cuda());
    assert(gaussGradDirs.is_contiguous() && gaussGradDirs.is_cuda());

    torch::Tensor directionHistogram =torch::zeros({keypoints.size(0), (int)(360/granularity)+1}, torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));

    gaussGradDirs = (gaussGradDirs/granularity).to(torch::kInt);

    torch::Tensor scales = scaleFactor * initialSigma * torch::exp2(keypoints.index({Slice(),2})/numIntervals);
    torch::Tensor radii = 4*scales;
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
    keyPointsWithGradientsCUDA<<<1, keypoints.size(0)>>>(devPitchedPtr, directionHistogramAccessor, radiusAccessor, keypointAccessor, gradMagsAccessor, gradDirsAccessor, gaussGradMags.size(2),gaussGradMags.size(1));
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    
    return directionHistogram;
}