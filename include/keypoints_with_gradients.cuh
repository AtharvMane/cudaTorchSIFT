#include <torch/torch.h>
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

void batchImshow(torch::Tensor imgBatch, std::string type, bool normalize);

void coutTensorShape(torch::IntArrayRef sizes, std::string varName);

torch::Tensor cvImgToTorchTensor(cv::Mat img, torch::DeviceType device);

cv::Mat tensorToMatGray(torch::Tensor imgTensor, bool normalize);

cv::Mat tensorToMatColor(torch::Tensor imgTensor, bool normalize);

torch::Tensor getGaussianMultiKernel(size_t size, torch::Tensor sigmas, torch::DeviceType device);

torch::Tensor getGaussianKernel(size_t size, double sigma, torch::DeviceType device);

std::tuple<torch::Tensor,int> getGaussianScales(int numIntervals, double baseSigma , torch::DeviceType device );

torch::Tensor getGrads(torch::Tensor input, torch::DeviceType device);

torch::Tensor getInitExtremaLocs(torch::Tensor diffGauss);

torch::Tensor getImageBoundMask(torch::Tensor ls, torch::Tensor dogs);

torch::Tensor getCoordResponses(torch::Tensor coords,torch::Tensor gradients, torch::Tensor dogs);

torch::Tensor getKeypoints(torch::Tensor diffGauss, int octaveIndex, torch::DeviceType device);

void showKeypoints(cv::Mat img, torch::Tensor keypoints);

cv::Mat draw_angled_rec(double x0, double y0, double width, double height, double angle, cv::Mat img);

torch::Tensor keypointsWithGradientsAndDescriptors(at::Tensor keypoints, at::Tensor gaussGradMags, at::Tensor gaussGradDirs, int granularity, double initialSigma, double scaleFactor, int numIntervals);

__global__ void createWeightTiles(int size, cudaPitchedPtr devPitchedPtr, torch::PackedTensorAccessor64<double,2> directionHistogramAccessor, torch::PackedTensorAccessor64<double,1> radiusAccessor, torch::PackedTensorAccessor64<double,2> keypointAccessor, torch::PackedTensorAccessor64<double,3> gradMagsAccessor, torch::PackedTensorAccessor64<int,3> gradDirsAccessor, int xLim, int yLim);

__global__ void createHistogram(cudaPitchedPtr devPitchedPtr, torch::PackedTensorAccessor64<double,2> directionHistogramAccessor, torch::PackedTensorAccessor64<double,1> radiusAccessor, torch::PackedTensorAccessor64<double,2> keypointAccessor, torch::PackedTensorAccessor64<double,3> gradMagsAccessor, torch::PackedTensorAccessor64<int,3> gradDirsAccessor);
