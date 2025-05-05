cmake_minimum_required(VERSION 3.10)

project(hack)
list(APPEND CMAKE_PREFIX_PATH "/home/arun/dev/cppLibs/libtorch-cxx11-abi-shared-with-deps-2.7.0+cpu/libtorch/share/cmake/Torch")

find_package(OpenCV REQUIRED)

find_package(Torch REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}
"/home/arun/dev/cppLibs/libtorch-cxx11-abi-shared-with-deps-2.7.0+cpu/libtorch/include"

)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(hack
    src/grokGen.cpp
)
add_executable(hackYolo
    src/test.cpp
    src/detection/detector.cpp
)
target_link_libraries(hack ${OpenCV_LIBS} ${TORCH_LIBRARIES})
target_link_libraries(hackYolo ${OpenCV_LIBS} ${TORCH_LIBRARIES})