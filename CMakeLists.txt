cmake_minimum_required(VERSION 3.23)
project(TP_HSP CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(TP_HSP main.cu)

set_target_properties(TP_HSP PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
