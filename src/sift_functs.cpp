#include <c10/cuda/CUDAException.h>
#include <torch/torch.h>
#include <iostream>
#include <bits/stdc++.h>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgcodecs/imgcodecs.hpp>
#include <string>
#include <cuda_runtime.h>
#include "keypoints_with_gradients.h"
#include <math_constants.h>
#include <math.h>
namespace F = torch::nn::functional;
using namespace torch::indexing;

void coutTensorShape(torch::IntArrayRef sizes, std::string varName){
    std::cout<<varName<<": [";
    for(int i = 0; i<sizes.size();i++){
        std::cout<<sizes[i]<<",";
    }
    std::cout<<"]"<<std::endl;
}

torch::Tensor cvImgToTorchTensor(cv::Mat img, torch::DeviceType device){
    if(img.channels()==3){
        img.convertTo(img,CV_64FC3, 1.0f/255.0f);
    }else if(img.channels()==1){
        img.convertTo(img,CV_64FC1, 1.0f/255.0f);
    }else if(img.channels()==4){
        img.convertTo(img,CV_64FC4, 1.0f/255.0f);
    }
    
    torch::Tensor imgTensor = torch::from_blob(img.data,{img.rows,img.cols,img.channels()}, torch::TensorOptions().dtype(torch::kFloat64)).clone();
    return imgTensor.to(device).index({None}).permute({0,3,1,2});
}

cv::Mat tensorToMatGray(torch::Tensor imgTensor, bool normalize){
    if(imgTensor.sizes().size()>2){
        throw std::runtime_error("Expected a single channel input");
    }else if (imgTensor.dtype()==torch::kFloat64){
        torch::Tensor dispTensor = imgTensor;//.clone();
        auto sizes = dispTensor.sizes();
        cv::Mat dispImg = cv::Mat(sizes[0], sizes[1], CV_64FC1,dispTensor.data_ptr());
        if(normalize){
            cv::normalize(dispImg, dispImg, torch::min(imgTensor).item().toDouble(), torch::max(imgTensor).item().toDouble(), cv::NORM_MINMAX, CV_64FC1);
        }
        return dispImg;
    }else if (imgTensor.dtype()==torch::kBool){
        torch::Tensor dispTensor = imgTensor;//.clone();
        auto sizes = dispTensor.sizes();
        cv::Mat dispImg = cv::Mat(sizes[0], sizes[1], CV_8UC1,dispTensor.data_ptr());
        if(normalize){
            cv::normalize(dispImg, dispImg, -2, 2, cv::NORM_MINMAX, CV_8UC1);
        }
        return dispImg;
    }else{
        throw std::runtime_error("Unsupported Type for tensor to Mat gray");
    }
}

cv::Mat tensorToMatColor(torch::Tensor imgTensor, bool normalize){
    if(imgTensor.sizes()[0]!=3){
        throw std::runtime_error("Expected a 3 channel input");
    }else{
        auto sizes = imgTensor.sizes();
        torch::Tensor dispTensor = imgTensor.clone();
        cv::Mat dispImg = cv::Mat(sizes[1], sizes[2], CV_64FC3,dispTensor.data_ptr());
        return dispImg;
    }
}

torch::Tensor getGaussianKernel(size_t size, double sigma, torch::DeviceType device){
    if(size&1){
        int mid = size/2;
        torch::Tensor a = torch::arange(-mid,mid+1,torch::TensorOptions().device(device).dtype(torch::kFloat64));
        a = torch::exp(-torch::square(a)/sigma).index({None});

        return a.t().matmul(a);
    }else{
        throw std::runtime_error("GAUSSIAN Kernel MUST have a ODD size.");
    }
    
}

torch::Tensor getGaussianMultiKernel(size_t size, torch::Tensor sigmas, torch::DeviceType device){
    if(size&1){
        int mid = size/2;
        torch::Tensor a = torch::arange(-mid,mid+1,torch::TensorOptions().device(device).dtype(torch::kFloat64)).index({None}).repeat({sigmas.size(0), 1});
        a = torch::exp(-torch::square(a.t()/sigmas)/2);
        a = sigmas.reciprocal()*(1/CUDART_SQRT_2PI_HI)*a;
        a = a.t().index({Slice(),Slice(),None});
        a = torch::einsum("nij,nkj->nik", {a,a});//torch::exp(-torch::square(a)/sigmas).index({None});

        return a.index({Slice(None,None,None), None});
    }else{
        throw std::runtime_error("GAUSSIAN Kernel MUST have a ODD size.");
    }
}

std::tuple<torch::Tensor,int> getGaussianScales(int numIntervals = 3, double baseSigma = 1.6, torch::DeviceType device = torch::kCUDA){
    int size = (int)round(16.0*baseSigma+1)|1;
    int numImages = numIntervals+3;
    torch::Tensor sigmas = torch::arange(numImages, torch::TensorOptions().device(device));
    sigmas = torch::exp2(sigmas/numIntervals)*baseSigma;
    return {sigmas, size};

}

void batchImshow(torch::Tensor imgBatch, std::string type, bool normalize){
    for (int i = 0;i<imgBatch.size(0);i++){
        for (int j=0;j<imgBatch.size(1);j++){
            torch::Tensor img = imgBatch.index({i,j});
            img = img.cpu().clone();
            cv::Mat cvImg = tensorToMatGray(img, normalize);
            cv::imshow(type, cvImg);
            cv::waitKey();
        }
    }
}

torch::Tensor getInitExtremaLocs(torch::Tensor diffGauss){
    torch::Tensor diffGaussPadded = F::pad(diffGauss.permute({1,0,2,3}), F::PadFuncOptions({1,1,1,1,1,1}).value(__builtin_inff64()));
    torch::Tensor diffGaussMaxes = F::max_pool3d(diffGaussPadded,F::MaxPool3dFuncOptions({3,3,3}).stride(1)).permute({1,0,2,3});
    diffGaussPadded = F::pad(-diffGauss.permute({1,0,2,3}), F::PadFuncOptions({1,1,1,1,1,1}).value(__builtin_inff64()));
    torch::Tensor diffGaussMins = F::max_pool3d(diffGaussPadded,F::MaxPool3dFuncOptions({3,3,3}).stride(1)).permute({1,0,2,3});
    torch::Tensor diffGaussExtremas = ((diffGaussMaxes==diffGauss)+(diffGaussMins==-diffGauss));
    return torch::stack(torch::where(diffGaussExtremas.index({Slice(),0}))).t();
}

torch::Tensor getGrads(torch::DeviceType device){
    torch::Tensor grader = torch::zeros({3,3,3,3},torch::TensorOptions().dtype(torch::kFloat64).device(device));
    grader[0][1][1][0]=-0.5;
    grader[0][1][1][2]=0.5;
    grader[1][1][0][1]=-0.5;
    grader[1][1][2][1]=0.5;
    grader[2][0][1][1]=-0.5;
    grader[2][2][1][1]=0.5;
    return grader;
}

torch::Tensor getImageBoundMask(torch::Tensor ls, torch::Tensor dogs){
    return (ls.index({Slice(),0})>0)*(ls.index({Slice(),1})>=0)*(ls.index({Slice(),2})>=0)*(ls.index({Slice(),0})<dogs.size(0)-1)*(ls.index({Slice(),1})<dogs.size(2))*(ls.index({Slice(),2})<dogs.size(3));
}



torch::Tensor getCoordResponses(torch::Tensor coords,torch::Tensor gradients, torch::Tensor dogs){
  torch::Tensor xHats=coords%1;
  torch::Tensor z=coords-xHats;
  z=z.to(torch::kLong);
  return dogs.index({z.index({Slice(),0}), 0,z.index({Slice(),1}),z.index({Slice(),2})})+ 0.5*torch::einsum("ni,in->n",{coords,gradients.index({Slice(),z.index({Slice(),0}),z.index({Slice(),1}),z.index({Slice(),2})})});
}


torch::Tensor getKeypoints(torch::Tensor diffGauss, int octaveIndex, torch::DeviceType device){
    torch::Tensor diffGaussExtremaLocs = getInitExtremaLocs(diffGauss);
    torch::Tensor grader = getGrads(device);

    torch::Tensor diffGaussPadded = F::pad(diffGauss.permute({1,0,2,3}), F::PadFuncOptions({1,1,1,1,1,1}).mode(torch::kReflect));
    torch::Tensor grads =F::conv3d(diffGaussPadded, grader.index({Slice(),None}));
    torch::Tensor doubleGrads = F::conv3d(F::pad(grads,F::PadFuncOptions({1,1,1,1,1,1})).index({Slice(),None}), grader.index({Slice(),None}));
    doubleGrads = doubleGrads.permute({2,3,4,0,1});
    grads = grads.permute({1,2,3,0});
    torch::Tensor finalDiffGaussExtremaLocs = torch::empty({0,3}, torch::TensorOptions().device(device).dtype(torch::kFloat64));

    // int numIter = 10;

    for(int mnop=0;mnop<5;mnop++){
        torch::Tensor doubleGradsSelected = doubleGrads.index({diffGaussExtremaLocs.index({Slice(),0}),diffGaussExtremaLocs.index({Slice(),1}),diffGaussExtremaLocs.index({Slice(),2})});
        torch::Tensor detMask = doubleGradsSelected.det()!=0.0;
        doubleGradsSelected = doubleGradsSelected.index({detMask});

        diffGaussExtremaLocs = diffGaussExtremaLocs.index({detMask});
        torch::Tensor doubleGradsInv = doubleGradsSelected.inverse();
        torch::Tensor gradsSelected = grads.index({diffGaussExtremaLocs.index({Slice(),0}),diffGaussExtremaLocs.index({Slice(),1}),diffGaussExtremaLocs.index({Slice(),2})});
        torch::Tensor xHats = torch::einsum("nij,nj->ni", {doubleGradsInv, gradsSelected});
        diffGaussExtremaLocs = xHats + diffGaussExtremaLocs.to(torch::kFloat64);
        torch::Tensor finalMask = (torch::abs(xHats.index({Slice(),0}))<0.5)*(torch::abs(xHats.index({Slice(),0}))<0.5)*(torch::abs(xHats.index({Slice(),0}))<0.5);
        finalDiffGaussExtremaLocs = torch::cat({finalDiffGaussExtremaLocs, diffGaussExtremaLocs.index({finalMask})});
        diffGaussExtremaLocs = torch::round(diffGaussExtremaLocs.index({finalMask==0})).to(torch::kLong);
        torch::Tensor mask = getImageBoundMask(diffGaussExtremaLocs, diffGauss);
        diffGaussExtremaLocs = diffGaussExtremaLocs.index({mask});
    }
    torch::Tensor resp = getCoordResponses(finalDiffGaussExtremaLocs, grads.permute({3,0,1,2}), diffGauss);

    torch::Tensor l = (torch::abs(resp))>0.05;
    torch::Tensor ret = finalDiffGaussExtremaLocs.index({l});
    resp = resp.index({l});
    torch::Tensor ret_r=(ret-ret%1).to(torch::kLong);
    torch::Tensor hessian=doubleGrads.index({ret_r.index({Slice(),0}),ret_r.index({Slice(),1}),ret_r.index({Slice(),2}),Slice(1,None),Slice(1,None)});
    torch::Tensor tracesSqr = torch::square(hessian.index({"...",0,0})+hessian.index({"...",0,0}));
    torch::Tensor det=hessian.det();
    torch::Tensor r=((tracesSqr/det)<12.1)*(det>0);
    ret=ret.index({r});
    resp=resp.index({r});
    torch::Tensor octave_info = torch::full(ret.size(0), octaveIndex, torch::TensorOptions().device(device));
    // print(resp[:,None].shape,"\n", octave_info[:,None].shape,"\n", ret.shape)
    ret=torch::cat({resp.index({Slice(),None}), octave_info.index({Slice(),None}), ret}, 1);
    return ret;
}

cv::Mat draw_angled_rec(double x0, double y0, double width, double height, double angle, cv::Mat img){
    width*=2;
    height*=2;
    double _angle = angle * CUDART_PI_HI/ 180.0;
    double b = cos(_angle) * 0.5;
    double a = sin(_angle) * 0.5;
    cv::Point2i pt0 = {(int)(x0 - a * height - b * width),(int)(y0 + b * height - a * width)};
    cv::Point2i pt1 = {(int)(x0 + a * height - b * width),(int)(y0 - b * height - a * width)};
    cv::Point2i pt2 = {(int)(2 * x0 - pt0.x), (int)(2 * y0 - pt0.y)};
    cv::Point2i pt3 = {(int)(2 * x0 - pt1.x), (int)(2 * y0 - pt1.y)};


    cv::line(img, pt0, pt1, {0, 255, 0}, 1);
    cv::line(img, pt1, pt2, {0, 255, 0}, 1);
    cv::line(img, pt2, pt3, {0, 255, 0}, 1);
    cv::line(img, pt3, pt0, {0, 255, 0}, 1);

    return img;

}

void showKeypoints(cv::Mat img, torch::Tensor keypoints){
    keypoints = keypoints.cpu();
    cv::Mat imgCopy;
    img.copyTo(imgCopy);
    for(int i=0;i<keypoints.size(0);i++){
        imgCopy = draw_angled_rec(keypoints[i][4].item().toDouble(),keypoints[i][3].item().toDouble(),5*keypoints[i][2].item().toDouble(),5*keypoints[i][2].item().toDouble(), 0, imgCopy);
        // cv::circle(imgCopy, {keypoints[i][4].item().toDouble(),keypoints[i][3].item().toDouble()}, 5*keypoints[i][2].item().toDouble(), {0,255,0}, 1);
    }
    cv::imshow("keypoints",imgCopy);
    cv::waitKey();
}

int main(int argc, char** argv){
    torch::DeviceType device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    cv::Mat img = cv::imread("/home/atharvmane/Pictures/Screenshot from 2023-01-04 02-10-26-new(1).png", cv::IMREAD_UNCHANGED);
    torch::Tensor imgTensor = cvImgToTorchTensor(img, device);
    imgTensor = 0.2989 * imgTensor.index({Slice(),2}) + 0.5870 * imgTensor.index({Slice(),1}) + 0.1140 * imgTensor.index({Slice(),0});
    imgTensor = imgTensor.index({Slice(),None});
    
    
    torch::Tensor gaussSigmas;
    int gaussKernelSize;
    std::tie<torch::Tensor, int>(gaussSigmas,gaussKernelSize) =  getGaussianScales(3, 1.6, device);
    int gaussHalf = gaussKernelSize/2;
    torch::Tensor gaussianKernels = getGaussianMultiKernel(gaussKernelSize, gaussSigmas, device);
    imgTensor = F::pad(imgTensor, F::PadFuncOptions({gaussHalf,gaussHalf,gaussHalf,gaussHalf}).mode(torch::enumtype::kReflect()));
    torch::Tensor gaussBlurs = F::conv2d(imgTensor, gaussianKernels).permute({1,0,2,3});
    torch::Tensor diffGauss = (gaussBlurs.index({Slice(0,-1),"..."}))-(gaussBlurs.index({Slice(1,None),"..."}));
    

    torch::Tensor keypoints = getKeypoints(diffGauss, 0, device);

    // showKeypoints(img,keypoints);
    torch::Tensor gaussBlursPadded = F::pad(gaussBlurs,F::PadFuncOptions({1,1,1,1}).mode(torch::kReflect));
    torch::Tensor grader2D = torch::tensor({{{{0.,0.,0.},{-0.5,0.,0.5},{0.,0.,0.}}},{{{0.,-0.5,0.},{0.,0.,0.},{0.,0.5,0.}}}}, torch::TensorOptions().device(device).dtype(torch::kFloat64));
    torch::Tensor gaussGrads = torch::conv2d(gaussBlursPadded, grader2D);
    torch::Tensor gaussGradsMags = torch::sqrt(torch::square(gaussGrads.index({Slice(),0}))+torch::square(gaussGrads.index({Slice(),1})));
    torch::Tensor gaussGradsDirs = (torch::atan2(gaussGrads.index({Slice(),1}), gaussGrads.index({Slice(),0}))*180*CUDART_2_OVER_PI);
    
    coutTensorShape(gaussGradsMags.sizes(),"gaussGradsMags");
    coutTensorShape(gaussGradsDirs.sizes(),"gaussGradsDirs");


    torch::Tensor directionHistogram = keypointsWithGradients(keypoints, gaussGradsMags, gaussGradsDirs, 10, 1.6, 3, 3);
    std::cout<<directionHistogram<<std::endl;
    return 0;
}