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
namespace F = torch::nn::functional;
using namespace torch::indexing;

int main(int argc, char*[]){ 
    cv::Mat img;
    cv::VideoCapture camera = cv::VideoCapture(0);
    torch::DeviceType device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    int numOctaves = 4;
    torch::Tensor gaussSigmas;
    int gaussKernelSize;
    std::tie<torch::Tensor, int>(gaussSigmas,gaussKernelSize) =  getGaussianScales(3, 1.6, device);
    torch::Tensor gaussianKernels = getGaussianMultiKernel(gaussKernelSize, gaussSigmas, device);
    int gaussHalf = gaussKernelSize/2;
    std::chrono::time_point<std::chrono::high_resolution_clock> starttime,endtime;
    // torch::Tensor keypoints = torch::empty({0,5}, torch::TensorOptions().device(device).dtype(torch::kFloat64));
    torch::Tensor grader2D = torch::tensor({{{{0.,0.,0.},{-0.5,0.,0.5},{0.,0.,0.}}},{{{0.,-0.5,0.},{0.,0.,0.},{0.,0.5,0.}}}}, torch::TensorOptions().device(device).dtype(torch::kFloat64));

    while(true){
        torch::Tensor keypointGradients = torch::empty({0,6}, torch::TensorOptions().device(device).dtype(torch::kFloat64));

        starttime = std::chrono::high_resolution_clock::now();
        // img = cv::imread("/home/atharvmane/Pictures/Screenshot from 2023-01-04 02-10-26-new(1).png", cv::IMREAD_UNCHANGED);

        camera.read(img);
        torch::Tensor imgTensor = cvImgToTorchTensor(img, device);
        imgTensor = 0.2989 * imgTensor.index({Slice(),2}) + 0.5870 * imgTensor.index({Slice(),1}) + 0.1140 * imgTensor.index({Slice(),0});
        imgTensor = imgTensor.index({Slice(),None});
        std::vector<torch::Tensor> inputImgPyramid = {imgTensor};
        std::vector<torch::Tensor> gaussianPyramid;
        std::vector<torch::Tensor> diffGaussianPyramid;
        std::vector<torch::Tensor> gaussGradDirPyramid;
        std::vector<torch::Tensor> gaussGradMagPyramid;

        // torch::Tensor keypointGradients = torch::empty({0,1}, torch::TensorOptions().device(device).dtype(torch::kFloat64));
        torch::Tensor grader2D = torch::tensor({{{{0.,0.,0.},{-0.5,0.,0.5},{0.,0.,0.}}},{{{0.,-0.5,0.},{0.,0.,0.},{0.,0.5,0.}}}}, torch::TensorOptions().device(device).dtype(torch::kFloat64));

        for(int i = 0; i<numOctaves-1; i++){
            inputImgPyramid.push_back(F::interpolate(inputImgPyramid[i],F::InterpolateFuncOptions().scale_factor(std::vector<double>({0.5,0.5})).mode(torch::kBilinear)));
        }
        for(int i = 0; i<numOctaves; i++){
            imgTensor = F::pad(inputImgPyramid[i], F::PadFuncOptions({gaussHalf,gaussHalf,gaussHalf,gaussHalf}).mode(torch::enumtype::kReflect()));
            gaussianPyramid.push_back(F::conv2d(imgTensor, gaussianKernels).permute({1,0,2,3}));
            diffGaussianPyramid.push_back((gaussianPyramid[i].index({Slice(0,-1),"..."}))-(gaussianPyramid[i].index({Slice(1,None),"..."})));            
            torch::Tensor keypoints = getKeypoints(diffGaussianPyramid[i], i, device);
            if(keypoints.size(0)==0){
                break;
            }
            torch::Tensor gaussBlursPadded = F::pad(gaussianPyramid[i],F::PadFuncOptions({1,1,1,1}).mode(torch::kReflect));
            torch::Tensor gaussGrads = torch::conv2d(gaussBlursPadded, grader2D);
            torch::Tensor gaussGradsMags = torch::sqrt(torch::square(gaussGrads.index({Slice(),0}))+torch::square(gaussGrads.index({Slice(),1})));
            torch::Tensor gaussGradsDirs = (torch::atan2(gaussGrads.index({Slice(),1}), gaussGrads.index({Slice(),0}))*90*CUDART_2_OVER_PI+180);

            // coutTensorShape(keypoints.sizes(),"kps");
            torch::Tensor directionHistogram = keypointsWithGradientsAndDescriptors(keypoints, gaussGradsMags, gaussGradsDirs, 10, 1.6, 3, 3);
            torch::Tensor directionMask = std::get<0>(torch::max(directionHistogram,1)).index({Slice(),None});
            directionMask = ((0.8*directionMask).repeat({1,directionHistogram.size(1)})<directionHistogram);
            std::vector<torch::Tensor> direcs=torch::where(directionMask);
            torch::Tensor keypointsWithGradientsTensor = keypoints.index({direcs[0]});

            keypointsWithGradientsTensor = torch::cat({keypointsWithGradientsTensor,direcs[1].index({Slice(),None})*10},1);
            coutTensorShape(keypointGradients.sizes(),"full");
            coutTensorShape(keypointsWithGradientsTensor.sizes(),"partial");

            keypointGradients = torch::cat({keypointGradients,keypointsWithGradientsTensor});
            // keypointGradients = torch::cat({keypointGradients,direcs[0].index({Slice(),None})});
 
        }
        // for(int i = 0; i<numOctaves; i++){
        //     torch::Tensor gaussBlursPadded = F::pad(gaussianPyramid[i],F::PadFuncOptions({1,1,1,1}).mode(torch::kReflect));
        //     torch::Tensor gaussGrads = torch::conv2d(gaussBlursPadded, grader2D);
        //     torch::Tensor gaussGradsMags = torch::sqrt(torch::square(gaussGrads.index({Slice(),0}))+torch::square(gaussGrads.index({Slice(),1})));
        //     torch::Tensor gaussGradsDirs = (torch::atan2(gaussGrads.index({Slice(),1}), gaussGrads.index({Slice(),0}))*180*CUDART_2_OVER_PI);

        //     torch::Tensor directionHistogram = keypointsWithGradientsAndDescriptors(keypoints, gaussGradsMags, gaussGradsDirs, 10, 1.6, 3, 3);
        //     torch::Tensor directionMask = std::get<0>(torch::max(directionHistogram,1)).index({Slice(),None});
        //     directionMask = ((0.8*directionMask).repeat({1,directionHistogram.size(1)})<directionHistogram);
        //     std::vector<torch::Tensor> direcs=torch::where(directionMask);
        //     keypointGradients = torch::cat({keypointGradients,direcs[0].index({None})});
        
        // }
        // keypoints = torch::cat({keypoints, keypointGradients},1);


        endtime = std::chrono::high_resolution_clock::now();
        auto st = std::chrono::time_point_cast<std::chrono::nanoseconds>(starttime).time_since_epoch().count();
        auto et = std::chrono::time_point_cast<std::chrono::nanoseconds>(endtime).time_since_epoch().count();
        std::cout<<"runtime: "<<(et-st)/1000000000.0<<" fps: "<<1000000000.0/(et-st)<<std::endl;
        c10::cuda::CUDACachingAllocator::emptyCache();
        showKeypoints(img, keypointGradients);

        // for(int i=0;i<numOctaves;i++){
        //     batchImshow(diffGaussianPyramid[i]*10, "impPyr", true);
        // }
    }
}
