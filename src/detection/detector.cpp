#include "detector.h"
namespace detector {

            transformer::transformer() = default;
            //Load a transformer model
            transformer::transformer(std::string_view model_path){
                model = torch::jit::load(model_path.data());
            }


            void transformer::pre_process_opencv_image(const cv::Mat& image){
                cv::resize(image, img_resized, cv::Size(800, 800));
                cv::cvtColor(img_resized, img_resized, cv::COLOR_BGR2RGB);
                img_resized.convertTo(img_resized, CV_32F, 1.0 / 255.0);
            }

            void transformer::normalize(){
                // Normalize using ImageNet mean and std
                std::vector<double> mean = {0.485, 0.456, 0.406};
                std::vector<double> std = {0.229, 0.224, 0.225};
                std::vector<cv::Mat> channels(3);
                cv::split(img_resized, channels);
                for (int i = 0; i < 3; ++i)
                    channels[i] = (channels[i] - mean[i]) / std[i];
                cv::merge(channels, img_resized);
            }
            void transformer::infer(){
                //Convert image to tensor
                auto input_tensor = torch::from_blob(img_resized.data, {1, 800, 800, 3}).permute({0, 3, 1, 2}).contiguous();
                input_tensor = input_tensor.to(torch::kF32);

                    // Run inference
                auto outputs = model.forward({input_tensor}).toTuple();
                auto pred_logits = outputs->elements()[0].toTensor(); // [1, 100, 91]
                auto pred_boxes = outputs->elements()[1].toTensor();  // [1, 100, 4] - normalized cx,cy,w,h

                auto logits = pred_logits[0]; // [100, 91]
                auto boxes = pred_boxes[0];   // [100, 4]

            }
            std::tuple<at::Tensor,at::Tensor> transformer::run_inference(cv::Mat image){
                pre_process_opencv_image(image);
                normalize();
                infer();
                return std::tuple<at::Tensor,at::Tensor>(logits,boxes);

            }

    
    void draw_boxes(const at::Tensor& boxes,const at::Tensor& logits,cv::Mat& image){
        float conf_thresh = 0.7;
        for (int i = 0; i < logits.size(0); ++i) {
            auto scores = logits[i].softmax(-1);
            auto max_score = scores.max(/*dim=*/0);
            int label = std::get<1>(max_score).item<int>();
            float score = std::get<0>(max_score).item<float>();
    
            if (label == 91 || score < conf_thresh) continue; // 91 = no-object class
    
            auto box = boxes[i]; // [cx, cy, w, h] normalized
            float cx = box[0].item<float>() * image.cols;
            float cy = box[1].item<float>() * image.rows;
            float w  = box[2].item<float>() * image.cols;
            float h  = box[3].item<float>() * image.rows;
    
            int x1 = cx - w / 2;
            int y1 = cy - h / 2;
            int x2 = cx + w / 2;
            int y2 = cy + h / 2;
    
            cv::rectangle(image, {x1, y1}, {x2, y2}, {0, 255, 0}, 2);
            cv::putText(image, std::to_string(label), {x1, y1 - 10}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 1);
        }
    }
}