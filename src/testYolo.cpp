#include <opencv2/core/types.hpp>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <vector>

torch::Tensor preprocess(const cv::Mat& image) {
    cv::Mat img;
    cv::resize(image, img, cv::Size(640, 640)); // resize to model input
    img.convertTo(img, CV_32F, 1.0 / 255.0);    // scale to [0,1]
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);  // BGR to RGB

    auto img_tensor = torch::from_blob(img.data, {1, 640, 640, 3}).permute({0, 3, 1, 2});
    return img_tensor.clone(); // clone needed to avoid memory issues

}
std::vector<std::string> class_names = {"gunnyBags"};

void draw_detections(cv::Mat& image, const at::Tensor& detections, float conf_thres = 0.3) {
    auto det = detections.accessor<float, 2>();

    for (int i = 0; i < detections.size(0); ++i) {
        float confidence = det[i][4];
        if (confidence < conf_thres) continue;

        int class_id = static_cast<int>(det[i][5]);
        float x1 = det[i][0];
        float y1 = det[i][1];
        float x2 = det[i][2];
        float y2 = det[i][3];

        cv::Rect box(cv::Point(x1, y1), cv::Point(x2, y2));
        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);

        std::string label = class_names[class_id] + " " + cv::format("%.2f", confidence);
        cv::putText(image, label, cv::Point(x1, y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
    }
}
int main(int argc, const char* argv[]){
    // torch::jit::script::Module model = torch::jit::load("yolo/best.torchscript");
    torch::jit::script::Module model = torch::jit::load("deps/yolov5s.torchscript");
    model.eval();
    if(argc < 2) 
        return 0;
    // Load the YOLOv5 TorchScript model

    
    // Load image using OpenCV
    cv::Mat image = cv::imread(argv[1]);
    if (image.empty()) {
        std::cerr << "Image not found!\n";
        return -1;
    }
    
        // Preprocess image
    torch::Tensor input_tensor = preprocess(image);
    
        // Inference
    std::vector<torch::jit::IValue> inputs = {input_tensor};
    auto output = model.forward(inputs).toTuple();
    
        // Post-process detections (optional)
    auto detections = output->elements()[0].toTensor().squeeze(0); // shape: (num_detections, 6)
    draw_detections(image,detections,std::stof(argv[2]));
    std::cout << "Detections:\n" << detections << std::endl;
    cv::imshow("YOLOv5 Inference", image);
    cv::waitKey(0);

    return 0;
}