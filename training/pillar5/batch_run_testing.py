import os
import glob
from pillar_5_forensics import analyze_shadows

def process_all_test_images():
    testing_dir = os.path.join(os.path.dirname(__file__), "testing")
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Supported image extensions
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_files = []
    for ext in valid_exts:
        image_files.extend(glob.glob(os.path.join(testing_dir, ext)))
        
    image_files = sorted(list(set(image_files)))
    print(f"Found {len(image_files)} images in {testing_dir}")
    
    summary = []
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_sub_dir = os.path.join(results_dir, base_name)
        print(f"\n================ Processing: {os.path.basename(img_path)} ================")
        try:
            analyze_shadows(img_path, output_dir=out_sub_dir, use_ml=True)
            summary.append((os.path.basename(img_path), "SUCCESS", out_sub_dir))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            summary.append((os.path.basename(img_path), f"ERROR: {e}", out_sub_dir))
            
    print("\n\n================ BATCH PROCESSING SUMMARY ================")
    for img_name, status, path in summary:
        print(f"Image: {img_name} -> {status}")
        if status == "SUCCESS":
            print(f"  Artifacts saved to: {path}")

if __name__ == "__main__":
    process_all_test_images()
