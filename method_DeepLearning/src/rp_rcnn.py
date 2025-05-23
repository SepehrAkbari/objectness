import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
# from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
import cv2
import os
import pandas as pd
from PIL import Image

def create_output_dirs(base_output_path):
    crops_path = os.path.join(base_output_path, "crops")
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)
    if not os.path.exists(crops_path):
        os.makedirs(crops_path)
    return crops_path

def get_faster_rcnn_model(device):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    # weights = fasterrcnn_resnet50_fpn.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    model.to(device)
    model.eval()
    return model

def main():
    print("Starting...")
    paintings_dir = "../../images/paintings/"
    output_base_dir = "../output/"
    
    output_crops_dir = create_output_dirs(output_base_dir)
    output_csv_path = os.path.join(output_base_dir, "data_frcnn.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    score_threshold = 0.5
    nms_iou_threshold = 0.3
    proposals_num = 15

    try:
        model = get_faster_rcnn_model(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("If using an older torchvision, try changing how the model is loaded.")
        return

    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])

    all_crop_info = []
    processed_image_count = 0

    if not os.path.exists(paintings_dir):
        print(f"Error: Paintings directory not found at {os.path.abspath(paintings_dir)}")
        return

    image_files = [f for f in os.listdir(paintings_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_filename in image_files:
        image_path = os.path.join(paintings_dir, image_filename)
        print(f"Processing: {image_filename}")

        try:
            img_pil = Image.open(image_path).convert("RGB")
            img_tensor = preprocess(img_pil).to(device)

            img_cv = cv2.imread(image_path)
            if img_cv is None:
                print(f"  Warning: Could not read image {image_filename} with OpenCV.")
                continue

            with torch.no_grad():
                predictions = model([img_tensor])

            pred = predictions[0]
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']

            keep_by_score = pred_scores > score_threshold
            filtered_boxes = pred_boxes[keep_by_score]
            filtered_scores = pred_scores[keep_by_score]

            if filtered_boxes.shape[0] == 0:
                print(f"  No proposals above score threshold {score_threshold} for {image_filename}")
                continue

            keep_by_nms_indices = torchvision.ops.nms(filtered_boxes, filtered_scores, nms_iou_threshold)
            
            final_boxes_tensor = filtered_boxes[keep_by_nms_indices]

            num_to_crop = min(final_boxes_tensor.shape[0], proposals_num)
            
            if num_to_crop == 0:
                print(f"  No proposals after NMS for {image_filename}")
                continue
                
            print(f"  Found {final_boxes_tensor.shape[0]} proposals after NMS, taking top {num_to_crop}")

            final_boxes = final_boxes_tensor.cpu().numpy().astype(int)

            is_wrong_file = "_WRONG" in image_filename.upper()
            base_filename_for_crop = os.path.splitext(image_filename)[0]

            for i in range(num_to_crop):
                box = final_boxes[i]
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                h, w, _ = img_cv.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    continue

                crop_img = img_cv[y1:y2, x1:x2]
                
                crop_filename = os.path.join(output_crops_dir, f"{base_filename_for_crop}_frcnn_crop{i}.jpg")
                cv2.imwrite(crop_filename, crop_img)

                all_crop_info.append({
                    'file_name': image_filename,
                    'crop_index': i,
                    'top_left_x': x1,
                    'top_left_y': y1,
                    'top_right_x': x2,
                    'top_right_y': y1,
                    'bottom_left_x': x1,
                    'bottom_left_y': y2,
                    'bottom_right_x': x2,
                    'bottom_right_y': y2,
                    'WRONG': is_wrong_file
                })
            processed_image_count +=1

        except Exception as e:
            print(f"  Error processing {image_filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_crop_info:
        df = pd.DataFrame(all_crop_info)
        df.to_csv(output_csv_path, index=False)
        print(f"\nProcessing complete. {processed_image_count} images processed successfully.")
        print(f"Cropped images saved to: {os.path.abspath(output_crops_dir)}")
        print(f"CSV metadata saved to: {os.path.abspath(output_csv_path)}")
    else:
        print("\nNo crops were generated.")

if __name__ == "__main__":
    print("Starting script...")
    main()