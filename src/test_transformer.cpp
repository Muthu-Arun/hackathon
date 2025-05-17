#include "detection/detector.h"
#include <ATen/core/TensorBody.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <tuple>
#include <chrono>

void test(cv::Mat& image){
    detector::transformer model("deps/detr_scripted.pt");
    std::tuple<at::Tensor, at::Tensor> output = model.run_inference(image);
    detector::draw_boxes(output,image);
}
void test(cv::Mat&& image){
    detector::transformer model("deps/detr_scripted.pt");
    std::tuple<at::Tensor, at::Tensor> output = model.run_inference(image);
    detector::draw_boxes(output,image);
    cv::imshow("Test Image",image);
    cv::waitKey();
}


int main(){
    cv::VideoCapture video("deps/Crowd walking on street.mp4");
    cv::Mat frame;
    test(cv::imread("deps/warehouse.jpeg"));
    std::cin.get();
    video >> frame;
    // cv::namedWindow("Video");
    while(!frame.empty()){
        std::cout << "Inside Loop\n";
        test(frame);
        video >> frame;
        cv::imshow("Video",frame);
        cv::waitKey(0);

    }
}