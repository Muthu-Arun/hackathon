#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>

int main() {
    torch::jit::script::Module model = torch::jit::load("deps/yolov5s.torchscript");
    model.eval();

    // Load and preprocess image
    cv::Mat image = cv::imread("deps/room.jpeg");
    if (image.empty()) {
        std::cerr << "Could not load image.\n";
        return -1;
    }

    cv::Mat img_resized;
    cv::resize(image, img_resized, cv::Size(640, 640));
    cv::cvtColor(img_resized, img_resized, cv::COLOR_BGR2RGB);
    img_resized.convertTo(img_resized, CV_32F, 1.0 / 255);

    auto input_tensor = torch::from_blob(img_resized.data, {1, 640, 640, 3}).permute({0, 3, 1, 2}).contiguous();
    input_tensor = input_tensor.to(torch::kF32);

    at::Tensor output = model.forward({input_tensor}).toTuple()->elements()[0].toTensor().squeeze(0);

    float conf_threshold = 0.4;

    for (int i = 0; i < output.size(0); ++i) {
        auto row = output[i];

        float objectness = row[4].item<float>();
        if (objectness < conf_threshold) continue;

        // Get class scores and max class
        at::Tensor class_scores = row.slice(0, 5);
        auto max_result = class_scores.max(0);
        float max_score = std::get<0>(max_result).item<float>();
        int class_id = std::get<1>(max_result).item<int>();

        float confidence = objectness * max_score;
        if (confidence < conf_threshold) continue;

        // Get bbox in cxcywh and convert to xyxy
        float cx = row[0].item<float>() * image.cols / 640.0;
        float cy = row[1].item<float>() * image.rows / 640.0;
        float w  = row[2].item<float>() * image.cols / 640.0;
        float h  = row[3].item<float>() * image.rows / 640.0;

        float x1 = cx - w / 2;
        float y1 = cy - h / 2;
        float x2 = cx + w / 2;
        float y2 = cy + h / 2;

        // Draw
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), {0, 255, 0}, 2);
        cv::putText(image, std::to_string(class_id), cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 1);
    }

    cv::imshow("Detection", image);
    cv::waitKey(0);
    return 0;
}
