import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms
import cv2 # OpenCV
import os
import pandas as pd
from PIL import Image
import argparse
import sys # For stderr and stdout control

def create_output_dirs_if_needed(base_output_path):
    """Creates output directories if they don't exist."""
    crops_path = os.path.join(base_output_path, "crops")
    if not os.path.exists(base_output_path): # Should be created by orchestrator
        try:
            os.makedirs(base_output_path) 
        except OSError as e:
            print(f"FRCNN_PY Error creating base temp directory {base_output_path}: {e}", file=sys.stderr)
            return None
    if not os.path.exists(crops_path):
        try:
            os.makedirs(crops_path)
        except OSError as e:
            print(f"FRCNN_PY Error creating crops directory {crops_path}: {e}", file=sys.stderr)
            return None 
    return crops_path

def get_faster_rcnn_model(device):
    """Loads a pre-trained Faster R-CNN model."""
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    return model

def main(input_image_path_arg, temp_output_dir_arg):
    output_crops_dir = create_output_dirs_if_needed(temp_output_dir_arg)
    if output_crops_dir is None:
        print(0) # CRITICAL: Output 0 to stdout for orchestrator
        sys.exit(1) # Exit with error

    temp_meta_csv_path = os.path.join(temp_output_dir_arg, "frcnn_meta.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    score_threshold = 0.5
    nms_iou_threshold = 0.3
    initial_proposals_limit = 30 # FRCNN can generate more, orchestrator will pick top 20 if needed

    try:
        model = get_faster_rcnn_model(device)
    except Exception as e:
        print(f"FRCNN_PY Error loading model: {e}", file=sys.stderr)
        print(0) 
        sys.exit(1)

    preprocess = transforms.Compose([transforms.ToTensor()])
    
    image_filename = os.path.basename(input_image_path_arg)
    base_filename_for_crop = os.path.splitext(image_filename)[0]
    saved_proposals_count = 0
    
    try:
        with open(temp_meta_csv_path, 'w') as meta_file:
            meta_file.write("relative_crop_path,x,y,width,height,score\n") # Header

            try:
                img_pil = Image.open(input_image_path_arg).convert("RGB")
            except FileNotFoundError:
                print(f"FRCNN_PY Error: Image file not found: {input_image_path_arg}", file=sys.stderr)
                print(0)
                sys.exit(1)
            
            img_tensor = preprocess(img_pil).to(device)
            img_cv = cv2.imread(input_image_path_arg)

            if img_cv is None:
                print(f"FRCNN_PY Warning: Could not read image {image_filename} with OpenCV.", file=sys.stderr)
                print(0)
                sys.exit(1)

            with torch.no_grad():
                predictions = model([img_tensor])

            pred = predictions[0]
            pred_boxes_tensor = pred['boxes']
            pred_scores_tensor = pred['scores']

            keep_by_score_indices = torch.where(pred_scores_tensor > score_threshold)[0]
            filtered_boxes_tensor = pred_boxes_tensor[keep_by_score_indices]
            filtered_scores_tensor = pred_scores_tensor[keep_by_score_indices]

            if filtered_boxes_tensor.shape[0] == 0:
                print(0)
                sys.exit(0) # Normal exit, just no proposals

            keep_by_nms_indices = torchvision.ops.nms(filtered_boxes_tensor, filtered_scores_tensor, nms_iou_threshold)
            
            # Take top N *after* NMS, sorted by score (NMS output is not guaranteed to be sorted by score,
            # but the input 'filtered_scores_tensor' was. Typically, nms implementations handle this,
            # or one might need to re-index scores and sort if taking a subset from NMS output).
            # For simplicity here, we assume the order from NMS is good enough for 'initial_proposals_limit'.
            final_indices_to_consider = keep_by_nms_indices[:initial_proposals_limit]

            final_boxes_tensor_cpu = filtered_boxes_tensor[final_indices_to_consider].cpu()
            final_scores_tensor_cpu = filtered_scores_tensor[final_indices_to_consider].cpu()

            if final_boxes_tensor_cpu.shape[0] == 0:
                print(0)
                sys.exit(0)
                            
            final_boxes_np = final_boxes_tensor_cpu.numpy().astype(int)
            final_scores_np = final_scores_tensor_cpu.numpy()

            for i in range(final_boxes_np.shape[0]):
                box = final_boxes_np[i]
                score = final_scores_np[i]
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                h_img, w_img, _ = img_cv.shape
                x1_c, y1_c = max(0, x1), max(0, y1) 
                x2_c, y2_c = min(w_img, x2), min(h_img, y2)
                
                width = x2_c - x1_c
                height = y2_c - y1_c

                if width <= 0 or height <= 0:
                    continue

                crop_img = img_cv[y1_c:y2_c, x1_c:x2_c]
                
                relative_crop_name = f"crops/{base_filename_for_crop}_frcnn_temp_crop{i}.jpg"
                temp_crop_filepath = os.path.join(temp_output_dir_arg, relative_crop_name)
                
                cv2.imwrite(temp_crop_filepath, crop_img)
                
                meta_file.write(f"{relative_crop_name},{x1_c},{y1_c},{width},{height},{score:.4f}\n")
                saved_proposals_count += 1
        
        print(saved_proposals_count) # CRITICAL: Print actual count to stdout

    except Exception as e:
        print(f"FRCNN_PY Error processing {image_filename}: {e}", file=sys.stderr)
        print(0) 
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FRCNN single image proposal generator.")
    parser.add_argument("input_image_path", type=str, help="Path to the input image.")
    parser.add_argument("temp_output_dir", type=str, help="Path to the temporary output directory for this image's results.")
    args = parser.parse_args()
    
    main(args.input_image_path, args.temp_output_dir)