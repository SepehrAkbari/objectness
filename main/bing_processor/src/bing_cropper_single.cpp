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

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "BING_CPP Usage: " << argv[0] << " <input_image_path> <num_proposals_to_generate> <temp_output_dir_for_this_image>" << std::endl;
        return 1;
    }

    std::string input_image_path_str = argv[1];
    int num_proposals_to_generate = 0;
    try {
        num_proposals_to_generate = std::stoi(argv[2]);
    } catch (const std::exception& e) {
        std::cerr << "BING_CPP Error: Invalid number of proposals: " << argv[2] << " - " << e.what() << std::endl;
        return 1;
    }
    std::string temp_output_dir_str = argv[3];
    std::string model_base_path = "../src/models/"; 

    std::filesystem::path currentCWD = fs::current_path();
    std::cerr << "BING_CPP Debug: Current actual CWD: " << currentCWD.string() << std::endl;
    std::cerr << "BING_CPP Debug: Relative model_base_path used: " << model_base_path << std::endl;
    std::filesystem::path resolvedModelPath = currentCWD / model_base_path;
    std::cerr << "BING_CPP Debug: Resolved model path BING will attempt to use: " << fs::absolute(resolvedModelPath).lexically_normal().string() << std::endl;


    std::string temp_crops_output_path = temp_output_dir_str + "/crops";
    try {
        if (!fs::exists(temp_output_dir_str)) {
            fs::create_directories(temp_output_dir_str);
        }
        if (!fs::exists(temp_crops_output_path)) {
            fs::create_directories(temp_crops_output_path);
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "BING_CPP Error creating temporary directories: " << e.what() << std::endl;
        return 1;
    }
    
    std::string temp_meta_csv_path = temp_output_dir_str + "/bing_meta.csv";

    cv::Ptr<cv::saliency::ObjectnessBING> bing = cv::saliency::ObjectnessBING::create();
    if (!bing) {
        std::cerr << "BING_CPP Error: Could not create ObjectnessBING instance." << std::endl;
        return 1;
    }

    bing->setBase(2);
    bing->setW(8);
    bing->setNSS(2); 
    
    std::cerr << "BING_CPP Debug: Setting training path for OpenCV BING to: " << model_base_path << std::endl;
    try {
        bing->setTrainingPath(model_base_path); 
    } catch (const cv::Exception& e) {
        std::cerr << "BING_CPP Error: Exception during setTrainingPath: " << e.what() << std::endl;
        return 1;
    }
    std::cerr << "BING_CPP Debug: Training path set." << std::endl;


    std::ofstream meta_file(temp_meta_csv_path);
    if (!meta_file.is_open()) {
        std::cerr << "BING_CPP Error: Could not open temporary meta CSV for writing: " << temp_meta_csv_path << std::endl;
        return 1;
    }
    meta_file << "relative_crop_path,x,y,width,height\n"; 

    fs::path input_fs_path(input_image_path_str);
    std::string image_filename_str = input_fs_path.filename().string();
    std::string base_filename_for_crop = input_fs_path.stem().string();
    
    cv::Mat image = cv::imread(input_image_path_str);
    if (image.empty()) {
        std::cerr << "BING_CPP Error: Could not read image: " << input_image_path_str << std::endl;
        meta_file.close();
        return 1; 
    }

    std::vector<cv::Vec4i> bounding_boxes; 
    bool success = false;
    std::cerr << "BING_CPP Debug: Calling computeSaliency..." << std::endl;
    try {
        success = bing->computeSaliency(image, bounding_boxes); 
    } catch (const cv::Exception& e) {
        std::cerr << "BING_CPP Error: OpenCV exception during computeSaliency for " << image_filename_str << ": " << e.what() << std::endl;
        meta_file.close();
        return 1; 
    }
    std::cerr << "BING_CPP Debug: computeSaliency call returned: " << (success ? "true" : "false") << std::endl;


    if (!success) {
        std::cerr << "BING_CPP Warning: computeSaliency explicitly failed for " << image_filename_str << "." << std::endl;
        if(bing->empty()){ 
            std::cerr << "BING_CPP Warning: BING algorithm state is empty. Model loading likely failed. Check resolved model_base_path and files within." << std::endl;
        }
        meta_file.close();
        return 0; 
    }
    
    if (bounding_boxes.empty()) {
        std::cerr << "BING_CPP Info: No initial proposals found by computeSaliency for " << image_filename_str << std::endl;
        meta_file.close();
        return 0; 
    }
    std::cerr << "BING_CPP Debug: Found " << bounding_boxes.size() << " initial BING proposals." << std::endl;


    int actual_proposals_to_take = std::min((int)bounding_boxes.size(), num_proposals_to_generate);
    if (actual_proposals_to_take < 0) actual_proposals_to_take = 0;
    int saved_count = 0;
    std::cerr << "BING_CPP Debug: Will attempt to save " << actual_proposals_to_take << " proposals." << std::endl;


    for (int i = 0; i < actual_proposals_to_take; ++i) {
        cv::Vec4i box_params = bounding_boxes[i];
        cv::Rect cv_box(box_params[0], box_params[1], box_params[2], box_params[3]);
        cv_box &= cv::Rect(0, 0, image.cols, image.rows); 

        if (cv_box.width <= 0 || cv_box.height <= 0) {
            continue;
        }

        cv::Mat crop = image(cv_box);
        
        std::string relative_crop_name = "crops/" + base_filename_for_crop + "_bing_temp_crop" + std::to_string(i) + ".jpg";
        std::string temp_crop_filepath = temp_output_dir_str + "/" + relative_crop_name;
        
        try {
            cv::imwrite(temp_crop_filepath, crop);
        } catch (const cv::Exception& e) {
            std::cerr << "BING_CPP Error: Failed to write crop image " << temp_crop_filepath << " : " << e.what() << std::endl;
            continue; 
        }

        meta_file << relative_crop_name << ","
                  << cv_box.x << "," << cv_box.y << ","
                  << cv_box.width << "," << cv_box.height << "\n";
        saved_count++;
    }

    meta_file.close();
    std::cerr << "BING_CPP Info: Finished. Actually saved " << saved_count << " proposals for " << image_filename_str << std::endl; 
    return 0; 
}