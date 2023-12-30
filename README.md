# cudaTorchSIFT: A Fully GPU resident implementation of SIFT Feature detector

##  How to  use?

In the root of this repository execute the following:
```
mkdir build && cd build
cmake ..
make
./b.out
```

## Caution!!
The project is still in development phase and thus very unstable. The keypoint detection is not repeatable across frames. It is advisable to use OpenCV SIFT implemetation instead.
The Author provides no warranty of work.

## Development Plans
- [ ] Check errors in the detector to increase repeatability of keypoints.
- [ ] Implement a custom GPU Tensor for for increased control over data and operations.

## Custom Tensor Baseline
- [ ] Write converters to convert to and from torch::Tensor and cv::Mat for contiuous development.
- [ ] Replicate optimum indexing, permuting, reshaping and Slicing like torch::Tensor.
- [ ] Write functions like convolution for optimal GPU Execution.