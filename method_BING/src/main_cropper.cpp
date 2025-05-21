#include <opencv2/saliency.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

struct CropInfo {
    std::string original_filename;
    int crop_index;
    cv::Rect bounding_box;
    bool is_wrong;
};

int main(int argc, char** argv) {
    std::string images_path = "../../images/paintings";
    std::string output_base_path = "../output";
    std::string output_crops_path = output_base_path + "/bing_crops";
    std::string output_csv_path = output_base_path + "/bing_crops.csv";
    std::string model_base_path = "../src/weights/";

    if (!fs::exists(output_base_path)) {
        fs::create_directories(output_base_path);
    }
    if (!fs::exists(output_crops_path)) {
        fs::create_directories(output_crops_path);
    }

    cv::Ptr<cv::saliency::ObjectnessBING> bing = cv::saliency::ObjectnessBING::create();
    if (!bing) {
        std::cerr << "Error: Could not create ObjectnessBING instance." << std::endl;
        return -1;
    }

    bing->setBase(2);
    bing->setW(8);
    bing->setNSS(2);

    bing->setTrainingPath(model_base_path);

    std::cout << "OpenCV BING initialized." << std::endl;
    std::cout << "models from: " << model_base_path << std::endl;
    std::cout << "With Base: " << bing->getBase() 
              << ", W: " << bing->getW() 
              << ", NSS: " << bing->getNSS() << std::endl;

    std::vector<CropInfo> all_crop_info;
    std::ofstream csv_file(output_csv_path);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Could not open CSV" << std::endl;
        return -1;
    }
    csv_file << "file_name,crop_index,top_left_x,top_left_y,top_right_x,top_right_y,bottom_left_x,bottom_left_y,bottom_right_x,bottom_right_y,WRONG\n";

    int total_images_processed = 0;
    for (const auto& entry : fs::directory_iterator(images_path)) {
        if (entry.is_regular_file()) {
            std::string image_path_str = entry.path().string();
            std::string image_filename_str = entry.path().filename().string();

            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            if (extension != ".jpg" && extension != ".jpeg") {
                continue;
            }

            std::cout << "Processing: " << image_filename_str << std::endl;

            cv::Mat image = cv::imread(image_path_str);
            if (image.empty()) {
                std::cerr << "Warning: Could not read image: " << image_path_str << std::endl;
                continue;
            }

            std::vector<cv::Vec4i> bounding_boxes;
            std::cout << "  Calling computeSaliency for " << image_filename_str << std::endl;
            bool success = bing->computeSaliency(image, bounding_boxes);

            if (!success) {
                std::cerr << "  computeSaliency failed for " << image_filename_str << ". Model is not loading." << std::endl;
                if(bing->empty()){
                    std::cerr << "  BING algorithm state is empty, model loading likely failed." << std::endl;
                }
                continue;
            }

            if (bounding_boxes.empty()) {
                std::cout << "  No proposals found for " << image_filename_str << " (but computeSaliency was successful)." << std::endl;
                continue;
            }

            int num_proposals_to_take = std::min((int)bounding_boxes.size(), 15);

            bool is_wrong_file = (image_filename_str.rfind("_WRONG") != std::string::npos);
            std::string base_filename_for_crop = image_filename_str;
            if (base_filename_for_crop.length() > 4 && base_filename_for_crop.substr(base_filename_for_crop.length() - 4) == ".jpg") {
                 base_filename_for_crop = base_filename_for_crop.substr(0, base_filename_for_crop.length() - 4);
            } else if (base_filename_for_crop.length() > 5 && base_filename_for_crop.substr(base_filename_for_crop.length() - 5) == ".jpeg") {
                 base_filename_for_crop = base_filename_for_crop.substr(0, base_filename_for_crop.length() - 5);
            }

            for (int i = 0; i < num_proposals_to_take; ++i) {
                cv::Vec4i box = bounding_boxes[i];
                cv::Rect cv_box(box[0], box[1], box[2] - box[0], box[3] - box[1]);

                cv_box &= cv::Rect(0, 0, image.cols, image.rows);
                if (cv_box.width <= 0 || cv_box.height <= 0) continue;

                cv::Mat crop = image(cv_box);
                std::string crop_filename = output_crops_path + "/" + base_filename_for_crop + "_crop" + std::to_string(i) + ".jpg";
                cv::imwrite(crop_filename, crop);

                csv_file << image_filename_str << ","
                         << i << ","
                         << cv_box.x << "," << cv_box.y << ","
                         << cv_box.x + cv_box.width << "," << cv_box.y << ","
                         << cv_box.x << "," << cv_box.y + cv_box.height << ","
                         << cv_box.x + cv_box.width << "," << cv_box.y + cv_box.height << ","
                         << (is_wrong_file ? "TRUE" : "FALSE") << "\n";
            }
            total_images_processed++;
        }
    }

    csv_file.close();
    std::cout << "Processing complete. " << total_images_processed << " images processed." << std::endl;
    std::cout << "Cropped images saved to: " << output_crops_path << std::endl;
    std::cout << "CSV metadata saved to: " << output_csv_path << std::endl;

    return 0;
}