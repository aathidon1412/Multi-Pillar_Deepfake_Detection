import cv2
import numpy as np
import pandas as pd
import math
import os
import random
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def calculate_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0: return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)

def extract_features(image_path, ransac_iters=800, target_dim=800):
    """
    Extracts multi-domain physics & illumination forensic features with scale-normalization:
    1. Ray Convergence & Inlier Ratio (RANSAC Vanishing Point)
    2. Angular Dispersion / Variance (Circular statistics)
    3. Shadow Penumbra Chromatic Variance (CIELAB)
    4. Normalized Sensor Noise Residual (Laplacian Variance on standardized canvas)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None: return None
        
        h, w = img.shape[:2]
        
        # Scale normalization to prevent 4K / low-res scale discrepancy
        scale = target_dim / max(h, w)
        img_norm = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        # 1. Structural Preprocessing & Vector Extraction
        gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
        edges = cv2.Canny(thresh, 40, 120)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        
        # 2. LAB Chromatic Variance in Shadow Regions
        lab = cv2.cvtColor(img_norm, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        shadow_pix = l_chan < np.percentile(l_chan, 35)
        shadow_chroma_var = float(np.std(a_chan[shadow_pix]) + np.std(b_chan[shadow_pix]))
        
        # 3. Scale-normalized sensor residual noise
        lap_var = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
        
        if lines is None:
            return {
                'total_lines': 0,
                'max_inliers': 0,
                'inlier_ratio': 0.0,
                'angular_variance_deg': 90.0,
                'shadow_chroma_var': round(shadow_chroma_var, 4),
                'lap_var': round(lap_var, 4)
            }
            
        lines = lines.reshape(-1, 4).tolist()
        angles = [math.atan2(l[3]-l[1], l[2]-l[0]) for l in lines]
        angles_mod = np.mod(angles, np.pi)
        median_angle = np.median(angles_mod)
        
        filtered_lines = []
        for line, angle in zip(lines, angles_mod):
            diff = abs(angle - median_angle)
            diff = min(diff, np.pi - diff)
            if diff < 0.35:
                filtered_lines.append(line)
                
        if len(filtered_lines) < 2:
            return {
                'total_lines': len(lines),
                'max_inliers': 0,
                'inlier_ratio': 0.0,
                'angular_variance_deg': 90.0,
                'shadow_chroma_var': round(shadow_chroma_var, 4),
                'lap_var': round(lap_var, 4)
            }
            
        # RANSAC Vanishing Point & Inlier estimation
        max_inliers = 0
        best_inlier_lines = []
        
        iterations = min(ransac_iters, len(filtered_lines) * len(filtered_lines))
        for _ in range(iterations):
            l1, l2 = random.sample(filtered_lines, 2)
            vp = calculate_intersection(l1, l2)
            if vp is None: continue
                
            inliers = 0
            inlier_lines = []
            for line in filtered_lines:
                den = math.sqrt((line[2]-line[0])**2 + (line[3]-line[1])**2)
                d = abs((line[2]-line[0])*(line[1]-vp[1]) - (line[0]-vp[0])*(line[3]-line[1])) / den if den > 0 else 9999
                if d < 50:
                    inliers += 1
                    inlier_lines.append(line)
                    
            if inliers > max_inliers:
                max_inliers = inliers
                best_inlier_lines = inlier_lines
                
        inlier_ratio = max_inliers / len(lines) if len(lines) > 0 else 0
        
        inlier_angles = [math.atan2(l[3]-l[1], l[2]-l[0]) for l in best_inlier_lines]
        mean_sin = np.mean([math.sin(2 * a) for a in inlier_angles]) if inlier_angles else 0
        mean_cos = np.mean([math.cos(2 * a) for a in inlier_angles]) if inlier_angles else 0
        R = math.sqrt(mean_sin**2 + mean_cos**2)
        angular_variance_deg = math.degrees((1.0 - R) * np.pi)
        
        return {
            'total_lines': len(lines),
            'max_inliers': max_inliers,
            'inlier_ratio': round(inlier_ratio, 4),
            'angular_variance_deg': round(angular_variance_deg, 4),
            'shadow_chroma_var': round(shadow_chroma_var, 4),
            'lap_var': round(lap_var, 4)
        }
    except Exception:
        return None

def process_image_wrapper(args):
    img_path, label = args
    feats = extract_features(img_path)
    if feats:
        feats['label'] = label
        feats['filename'] = Path(img_path).name
        return feats
    return None

def generate_dataset(dataset_dir, output_csv="pillar5_training_dataset.csv", samples_per_class=1200):
    dataset_dir = Path(dataset_dir)
    authentic_dir = dataset_dir / "authentic"
    fake_dir = dataset_dir / "fake"
    
    tasks = []
    
    # 1. Collect Authentic images
    if authentic_dir.exists():
        auth_images = [str(p) for p in authentic_dir.rglob("*.*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        random.seed(42)
        random.shuffle(auth_images)
        auth_sample = auth_images[:samples_per_class]
        print(f"Sampled {len(auth_sample)} Authentic Images.")
        for p in auth_sample:
            tasks.append((p, 1))
            
    # 2. Collect Balanced Fake images across all subfolders
    if fake_dir.exists():
        fake_subfolders = [f for f in fake_dir.iterdir() if f.is_dir()]
        fake_sample = []
        if fake_subfolders:
            per_folder = max(1, samples_per_class // len(fake_subfolders))
            for sub in fake_subfolders:
                sub_images = [str(p) for p in sub.rglob("*.*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                random.seed(42)
                random.shuffle(sub_images)
                selected = sub_images[:per_folder]
                print(f"Sampled {len(selected)} Fake images from {sub.name}")
                fake_sample.extend(selected)
        else:
            sub_images = [str(p) for p in fake_dir.rglob("*.*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            random.seed(42)
            random.shuffle(sub_images)
            fake_sample = sub_images[:samples_per_class]
            
        print(f"Sampled total {len(fake_sample)} Fake Images across all generators.")
        for p in fake_sample:
            tasks.append((p, 0))
            
    print(f"Total dataset extraction tasks: {len(tasks)}. Running multi-core extraction...")
    
    data = []
    num_workers = min(multiprocessing.cpu_count(), 8)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, result in enumerate(executor.map(process_image_wrapper, tasks)):
            if result:
                data.append(result)
            if (idx + 1) % 200 == 0 or idx + 1 == len(tasks):
                print(f"Extracted features for {idx + 1}/{len(tasks)} images...")
                
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"\n=======================================================")
    print(f"Training dataset created successfully with {len(df)} samples!")
    print(f"Authentic samples: {len(df[df['label']==1])}")
    print(f"Fake (AI) samples: {len(df[df['label']==0])}")
    print(f"Saved to: {output_csv}")
    print(f"=======================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="dataset", help="Directory containing 'authentic' and 'fake' subfolders")
    parser.add_argument("--output", default="pillar5_training_dataset.csv")
    parser.add_argument("--samples", type=int, default=1200, help="Number of images per class")
    args = parser.parse_args()
    generate_dataset(args.dir, args.output, args.samples)
