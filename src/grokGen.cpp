#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>

// Function to preprocess image for LibTorch model
torch::Tensor preprocess_image(const cv::Mat& image) {
    // Resize image to 224x224 (common input size for many models)
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(224, 224));
    
    // Convert BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    
    // Convert to float and normalize to [0,1]
    rgb.convertTo(rgb, CV_32F, 1.0/255.0);
    
    // Convert to tensor
    auto tensor = torch::from_blob(rgb.data, {1, rgb.rows, rgb.cols, 3});
    
    // Permute to [1,3,224,224] (batch, channels, height, width)
    tensor = tensor.permute({0, 3, 1, 2});
    
    // Normalize with ImageNet means and stds
    tensor = tensor.sub_(torch::tensor({0.485, 0.456, 0.406}).view({1, 3, 1, 1}));
    tensor = tensor.div_(torch::tensor({0.229, 0.224, 0.225}).view({1, 3, 1, 1}));
    
    return tensor;
}

int main() {
    try {
        // === OpenCV Demo: Basic Image Processing ===
        // Load image using OpenCV
        cv::Mat image = cv::imread("docs/AI Hackathon Challenge 2025.jpeg");
        if (image.empty()) {
            throw std::runtime_error("Could not load image");
        }
        
        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        
        // Apply edge detection
        cv::Mat edges;
        cv::Canny(gray, edges, 100, 200);
        
        // Save processed images
        cv::imwrite("grayscale.jpg", gray);
        cv::imwrite("edges.jpg", edges);
        
        // === LibTorch Demo: Image Classification ===
        // Load pre-trained model (e.g., ResNet18)
        torch::jit::script::Module model;
        try {
            model = torch::jit::load("resnet18.pt");
            model.eval();
        } catch (const c10::Error& e) {
            throw std::runtime_error("Error loading model: " + std::string(e.what()));
        }
        
        // Preprocess image
        torch::Tensor input_tensor = preprocess_image(image);
        
        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        auto output = model.forward(inputs).toTensor();
        
        // Get top-5 predictions
        auto [values, indices] = output.softmax(1).topk(5);
        
        // Print results (assuming ImageNet classes)
        std::cout << "Top-5 predictions:\n";
        for (int i = 0; i < 5; ++i) {
            std::cout << "Class " << indices[0][i].item<int>()
                      << ": " << values[0][i].item<float>() * 100 << "%\n";
        }
        
        // === OpenCV Demo: Real-time Video Processing ===
        cv::VideoCapture cap(0); // Open default camera
        if (!cap.isOpened()) {
            throw std::runtime_error("Could not open camera");
        }
        
        cv::Mat frame;
        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            
            // Apply real-time processing (e.g., face detection)
            cv::CascadeClassifier face_cascade;
            face_cascade.load("haarcascade_frontalface_default.xml");
            
            std::vector<cv::Rect> faces;
            cv::Mat frame_gray;
            cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
            face_cascade.detectMultiScale(frame_gray, faces);
            
            // Draw rectangles around faces
            for (const auto& face : faces) {
                cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
            }
            
            // Display frame
            cv::imshow("Camera", frame);
            if (cv::waitKey(1) == 27) break; // Exit on ESC
        }
        
        cv::destroyAllWindows();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}