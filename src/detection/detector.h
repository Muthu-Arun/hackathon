#include <ATen/core/TensorBody.h>
#include <string_view>
#include <torch/csrc/jit/api/module.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <iostream>
namespace detector {
    class transformer{

        public:
            transformer();
            //Load a transformer model
            transformer(std::string_view model_path);
            // Run inference to return logits and boxes in a std::pair
            std::tuple<at::Tensor,at::Tensor> run_inference(cv::Mat image);
        private:
            const std::vector<double> mean = {0.485, 0.456, 0.406};
            const std::vector<double> std = {0.229, 0.224, 0.225};
            cv::Mat img_resized;
            at::Tensor logits,boxes;
            // c10::intrusive_ptr<at::ivalue::Tuple> outputs;
            
        private:
            torch::jit::script::Module model;

            void pre_process_opencv_image(const cv::Mat& image);

            void normalize();
            void infer();
    };
    void draw_boxes(std::tuple<at::Tensor,at::Tensor>& output,cv::Mat& image);
    }
