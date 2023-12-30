cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app LANGUAGES CXX CUDA)
set(TORCH_PREFIX "/home/atharvmane/libtorch")
message("found pytorch prefix=${TORCH_PREFIX}")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PREFIX})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CAFFE2_USE_CUDNN ON)
set(CAFFE2_USE_CUSPARSELT ON)
find_package(Torch REQUIRED)
find_package(Caffe2 REQUIRED)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -DTORCH_USE_CUDA_DSA=1 -DTORCH_USE_CUDA_DSA=1")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TORCH_CUDA_FLAGS} -DTORCH_USE_CUDA_DSA=1 -DTORCH_USE_CUDA_DSA=1")
message("CXX FLAGS=${CMAKE_CXX_FLAGS}")

message("TORCH FLAGS=${TORCH_CXX_FLAGS}")

message("CUDA FLAGS=${CMAKE_CUDA_FLAGS}")
include_directories(include)
add_library(kpsCUDA STATIC src/keypoints_with_gradients.cu)
target_link_libraries(kpsCUDA ${TORCH_LIBRARIES})
add_library(kpsCPU SHARED src/keypoints_with_gradients_cpu.cu)
target_link_libraries(kpsCPU PUBLIC kpsCUDA)

add_library(siftFuncs src/sift_functs.cpp)

target_link_libraries(siftFuncs kpsCPU ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})

add_library(customTensor STATIC src/CustomCudaTensor.cu)
target_link_libraries(customTensor ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})


add_executable(b.out src/SIFT.cpp)

target_link_libraries(b.out siftFuncs kpsCPU ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})