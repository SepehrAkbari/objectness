#include <opencv2/saliency.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm> // For std::sort if needed
#include <filesystem> // For iterating through image files (C++17)
                     // Or use a platform-specific way or a library like dirent.h

namespace fs = std::filesystem;

struct CropInfo {
    std::string original_filename;
    int crop_index;
    cv::Rect bounding_box; // top_left_x, top_left_y, width, height
    bool is_wrong;
};

// Helper function to convert cv::Rect to your coordinate string if needed
// Or just store x, y, width, height and derive corners later.


int main(int argc, char** argv) {
    // --- Configuration ---
    std::string images_path = "../images"; // Path to your images folder
    std::string output_crops_path = "../output_bing_crops";
    std::string output_csv_path = "../output_bing_crops.csv";
    std::string model_base_path = "../src_bing_opencv/bing_model_opencv/"; // Relative to where executable runs or use absolute

    // Create output directory if it doesn't exist
    if (!fs::exists(output_crops_path)) {
        fs::create_directories(output_crops_path);
    }

    // --- Initialize BING ---
    cv::Ptr<cv::saliency::ObjectnessBING> bing = cv::saliency::ObjectnessBING::create();
    if (!bing) {
        std::cerr << "Error: Could not create ObjectnessBING instance." << std::endl;
        return -1;
    }

    // The setTrainingPath for OpenCV's BING usually points to the DIRECTORY
    // containing the .wS1.yml, .wS2.yml, and .idx.yml files.
    // The class internally constructs the full filenames based on its parameters
    // (base, W, color space).
    bing->setBase(2);
    bing->setW(8);
    bing->setNSS(2); // A common default for BING

    bing->setTrainingPath(model_base_path);

    // You might need to set parameters to match the model you chose, e.g., for MAXBGR
    // The defaults are often base=2, W=8, NSS=2. Color space choice is usually
    // handled by which model files it finds/loads based on internal naming conventions.
    // Or the OpenCV API might have a direct way to specify which model variant to use if multiple are in the path.
    // For now, let's assume it picks up one of the sets based on default parameters or
    // an internal mechanism if multiple are in the trainingPath.
    // If it doesn't work, we might need to explicitly point to one set, e.g. by putting only one set in trainingPath.

    std::cout << "OpenCV BING initialized." << std::endl;
    std::cout << "Attempting to use models from: " << model_base_path << std::endl;
    std::cout << "With Base: " << bing->getBase() 
              << ", W: " << bing->getW() 
              << ", NSS: " << bing->getNSS() << std::endl;

    std::vector<CropInfo> all_crop_info;
    std::ofstream csv_file(output_csv_path);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Could not open CSV file for writing." << std::endl;
        return -1;
    }
    csv_file << "file_name,crop_index,top_left_x,top_left_y,top_right_x,top_right_y,bottom_left_x,bottom_left_y,bottom_right_x,bottom_right_y,WRONG\n";


    // --- Process Images ---
    int total_images_processed = 0;
    for (const auto& entry : fs::directory_iterator(images_path)) {
        if (entry.is_regular_file()) {
            std::string image_path_str = entry.path().string();
            std::string image_filename_str = entry.path().filename().string();

            // Check for JPG extension
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
                std::cerr << "  computeSaliency failed for " << image_filename_str << ". This might indicate a model loading issue." << std::endl;
                // Check if the object is empty, which can happen if read() fails
                if(bing->empty()){
                    std::cerr << "  BING algorithm state is empty, model loading likely failed catastrophically." << std::endl;
                }
                continue;
            }

            if (bounding_boxes.empty()) {
                std::cout << "  No proposals found for " << image_filename_str << " (but computeSaliency was 'successful')." << std::endl;
                continue;
            }

            // BING proposals are often sorted by objectness value.
            // You want ~10-15 meaningful crops.
            int num_proposals_to_take = std::min((int)bounding_boxes.size(), 15); // Take top 15 or fewer

            bool is_wrong_file = (image_filename_str.rfind("_WRONG") != std::string::npos);
            std::string base_filename_for_crop = image_filename_str;
            if (base_filename_for_crop.length() > 4 && base_filename_for_crop.substr(base_filename_for_crop.length() - 4) == ".jpg") {
                 base_filename_for_crop = base_filename_for_crop.substr(0, base_filename_for_crop.length() - 4);
            } else if (base_filename_for_crop.length() > 5 && base_filename_for_crop.substr(base_filename_for_crop.length() - 5) == ".jpeg") {
                 base_filename_for_crop = base_filename_for_crop.substr(0, base_filename_for_crop.length() - 5);
            }


            for (int i = 0; i < num_proposals_to_take; ++i) {
                cv::Vec4i box = bounding_boxes[i]; // format seems to be [minX, minY, maxX, maxY]
                cv::Rect cv_box(box[0], box[1], box[2] - box[0], box[3] - box[1]);

                // Ensure box is within image bounds
                cv_box &= cv::Rect(0, 0, image.cols, image.rows);
                if (cv_box.width <= 0 || cv_box.height <= 0) continue;

                cv::Mat crop = image(cv_box);
                std::string crop_filename = output_crops_path + "/" + base_filename_for_crop + "_crop" + std::to_string(i) + ".jpg";
                cv::imwrite(crop_filename, crop);

                // Record for CSV
                // top_left, top_right, bottom_left, bottom_right
                // (x,y), (x+w, y), (x, y+h), (x+w, y+h)
                csv_file << image_filename_str << ","
                         << i << ","
                         << cv_box.x << "," << cv_box.y << ","                                 // top_left_x, top_left_y
                         << cv_box.x + cv_box.width << "," << cv_box.y << ","                 // top_right_x, top_right_y
                         << cv_box.x << "," << cv_box.y + cv_box.height << ","                // bottom_left_x, bottom_left_y
                         << cv_box.x + cv_box.width << "," << cv_box.y + cv_box.height << "," // bottom_right_x, bottom_right_y
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