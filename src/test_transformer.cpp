#include "detection/detector.h"
#include <ATen/core/TensorBody.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <tuple>
#include <chrono>

void test(cv::Mat& image){
    detector::transformer model("/home/arun/projects/hackathon/deps/detr_scripted.pt");
    std::tuple<at::Tensor, at::Tensor> output = model.run_inference(image);
    detector::draw_boxes(output,image);
}
void test(cv::Mat&& image){
    detector::transformer model("/home/arun/projects/hackathon/deps/detr_scripted.pt");
    std::tuple<at::Tensor, at::Tensor> output = model.run_inference(image);
    detector::draw_boxes(output,image);
    cv::imshow("Test Image",image);
    cv::waitKey();
}


int main(){
    cv::VideoCapture video("/home/arun/projects/hackathon/deps/Crowd walking on street.mp4"),output_video;
    cv::Mat frame;
    test(cv::imread("/home/arun/projects/hackathon/deps/warehouse.jpeg"));

    std::cin.get();
    while(video.isOpened()){
        std::cout << "Inside Loop\n";
        
        video >> frame;
        test(frame);
        
        cv::imshow("Video",frame);
        cv::waitKey(5);

    }
}