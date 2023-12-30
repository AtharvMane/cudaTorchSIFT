#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/torch.h>
#include <iostream>
#include <bits/stdc++.h>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgcodecs/imgcodecs.hpp>
#include <string>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <keypoints_with_gradients.cuh>

template <typename T> class customCudaTensor{
private:
std::vector<size_t>shape;
T* data;
std::string device;
public:
customCudaTensor<T>(){}
customCudaTensor<T>(T* dataptr, std::vector<size_t> shape){}

torch::Tensor toTorchTensor(){}
};

template <typename T> class customCudaTensor <T> fromCVMat(cv::Mat){

};
template <typename T> class customCudaTensor <T> fromTorchTensor(torch::Tensor){};