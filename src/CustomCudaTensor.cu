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


class customCudaTensor{
private:
std::vector<size_t>shape;
char* data; 
public:
customCudaTensor(){}

};