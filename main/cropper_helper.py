import pandas as pd
from skimage import io
import os

SOURCE_IMAGE_DIR = "../images/paintings/"
OUTPUT_CROPS_DIR = "output_crops/"
CSV_FILE_PATH = "output/combined_data.csv"

try:
    df = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"Error: The CSV file was not found at {CSV_FILE_PATH}")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

def parse_filename(original_filename_str):
    name_part = original_filename_str.replace(".jpg", "")
    
    parts = name_part.split('_')
    if len(parts) < 2:
        print(f"Warning: Filename '{original_filename_str}' format is unexpected. Could not parse.")
        return "UNKNOWN", "UNKNOWN", "0"

    num = parts[-1]
    if not num.isdigit():
        print(f"Warning: Last part of filename '{name_part}' is not a number. Using '0' as num.")
        num = "0"
        last_name = parts[-1] if len(parts) >=1 else "UNKNOWN"
        first_name = "_".join(parts[:-1]) if len(parts) > 1 else "UNKNOWN"
    else: 
        if len(parts) < 2:
             last_name = "UNKNOWN"
             first_name = "UNKNOWN"
        elif len(parts) == 2:
            last_name = ""
            first_name = parts[0]
        else:
            last_name = parts[-2]
            first_name = "_".join(parts[:-2])
            
    return first_name, last_name, num

def get_source_label(frcnn_source_val, bing_source_val):
    is_frcnn = str(frcnn_source_val).upper() == 'TRUE'
    is_bing = str(bing_source_val).upper() == 'TRUE'

    if is_frcnn:
        return "FRCNN"
    elif is_bing:
        return "BING"
    else:
        return "RANDOM"

if __name__ == '__main__':
    os.makedirs(OUTPUT_CROPS_DIR, exist_ok=True)
    print(f"Output directory: {os.path.abspath(OUTPUT_CROPS_DIR)}")

    grouped = df.groupby('original_filename')

    processed_crops_count = 0

    for img_name_in_csv, group in grouped:
        source_image_path = os.path.join(SOURCE_IMAGE_DIR, img_name_in_csv)
        
        print(f"\nProcessing original image: {source_image_path}...")
        try:
            image = io.imread(source_image_path)
        except FileNotFoundError:
            print(f"Error: Source image not found at {source_image_path}. Stopping execution.")
            exit()
        except Exception as e:
            print(f"Error loading image {source_image_path}: {e}. Skipping this image.")
            continue

        first_name, last_name, num_str = parse_filename(img_name_in_csv)

        for index, row in group.iterrows():
            crop_idx = row['crop_idx']

            y1, y2 = int(row['top_left_y']), int(row['bottom_right_y'])
            x1, x2 = int(row['top_left_x']), int(row['bottom_right_x'])

            if not (0 <= y1 < y2 <= image.shape[0] and 0 <= x1 < x2 <= image.shape[1]):
                print(f"  Warning: Invalid crop coordinates for {img_name_in_csv}, crop {crop_idx}: ({x1},{y1})-({x2},{y2}) for image size {image.shape}. Skipping this crop.")
                continue
                
            cropped_image = image[y1:y2, x1:x2]

            source_label = get_source_label(row['FRCNN_source'], row['BING_source'])

            if last_name:
                base_output_name = f"{first_name}_{last_name}_{num_str}_crop{crop_idx+1}_{source_label}"
            else:
                 base_output_name = f"{first_name}_{num_str}_crop{crop_idx+1}_{source_label}"

            if str(row['WRONG_file']).upper() == 'TRUE':
                output_filename = f"{base_output_name}_WRONG.jpg"
            else:
                output_filename = f"{base_output_name}.jpg"
            
            output_path = os.path.join(OUTPUT_CROPS_DIR, output_filename)
            
            try:
                io.imsave(output_path, cropped_image, quality=95)
                processed_crops_count += 1
                if processed_crops_count % 100 == 0 :
                    print(f"  Saved {output_filename} (Total crops: {processed_crops_count})")
            except Exception as e:
                print(f"  Error saving cropped image {output_path}: {e}")
        
    print(f"\nProcessing complete. Total crops saved: {processed_crops_count} / 25")