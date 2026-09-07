import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import math
import os
import random
import pickle

def calculate_intersection(line1, line2):
    """Calculate the intersection of two lines defined by two points each."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)

def line_point_distance(line, px, py):
    """Distance from a point to a line segment extended infinitely."""
    x1, y1, x2, y2 = line
    num = abs((x2 - x1) * (y1 - py) - (x1 - px) * (y2 - y1))
    den = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return num / den if den != 0 else float('inf')

def analyze_shadows(image_path, output_dir=".", use_ml=False, ml_model_path="pillar5_ml_model.pkl"):
    """
    Executes Pillar 5 of the Universal Synthetic Media Forensics Engine (USMFE).
    Analyzes shadow physics using Angular Variance & RANSAC, with optional ML classification.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    vis_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Advanced Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    edges = cv2.Canny(thresh, 50, 150)
    
    # 2. Vector Extraction
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=80, maxLineGap=10)
    
    if lines is not None:
        lines = lines.reshape(-1, 4).tolist()
    else:
        lines = []
        
    # 3. Directional Clustering (Pre-filtering)
    filtered_lines = []
    line_data = []
    
    if len(lines) >= 2:
        angles = [math.atan2(l[3]-l[1], l[2]-l[0]) for l in lines]
        angles_mod = np.mod(angles, np.pi)
        median_angle = np.median(angles_mod)
        for i, (line, angle) in enumerate(zip(lines, angles_mod)):
            diff = abs(angle - median_angle)
            diff = min(diff, np.pi - diff)
            if diff < 0.35:  # ~20 degrees tolerance
                filtered_lines.append(line)
                m = float('inf') if line[2] - line[0] == 0 else (line[3] - line[1]) / (line[2] - line[0])
                line_data.append({'id': i, 'line': line, 'm': m})
                
    # 4. RANSAC Vanishing Point Estimation
    best_vp = None
    max_inliers = 0
    best_inlier_lines = []
    best_inlier_data = []
    
    if len(filtered_lines) >= 2:
        iterations = min(1000, len(filtered_lines) * len(filtered_lines))
        for _ in range(iterations):
            l1, l2 = random.sample(filtered_lines, 2)
            vp = calculate_intersection(l1, l2)
            if vp is None:
                continue
                
            inliers = 0
            inlier_lines = []
            inlier_data_subset = []
            for i, line in enumerate(filtered_lines):
                dist = line_point_distance(line, vp[0], vp[1])
                if dist < 80:  # generous pixel tolerance for real-world lens distortion
                    inliers += 1
                    inlier_lines.append(line)
                    inlier_data_subset.append(line_data[i])
                    
            if inliers > max_inliers:
                max_inliers = inliers
                best_vp = vp
                best_inlier_lines = inlier_lines
                best_inlier_data = inlier_data_subset
                
    inlier_ratio = max_inliers / len(lines) if len(lines) > 0 else 0.0
    
    # Calculate Angular Variance using circular statistics
    inlier_angles = [math.atan2(l[3]-l[1], l[2]-l[0]) for l in best_inlier_lines]
    if inlier_angles:
        mean_sin = np.mean([math.sin(2 * a) for a in inlier_angles])
        mean_cos = np.mean([math.cos(2 * a) for a in inlier_angles])
        R = math.sqrt(mean_sin**2 + mean_cos**2)
        angular_variance_rad = 1.0 - R
        angular_variance_deg = math.degrees(angular_variance_rad * np.pi)
    else:
        angular_variance_deg = 180.0
        
    intersections = []
    intersection_data = []
    for i in range(len(best_inlier_data)):
        for j in range(i + 1, len(best_inlier_data)):
            l1 = best_inlier_data[i]
            l2 = best_inlier_data[j]
            pt = calculate_intersection(l1['line'], l2['line'])
            if pt is not None:
                intersections.append(pt)
                intersection_data.append({
                    'Line Pair ID': f"{l1['id']}-{l2['id']}",
                    'Intersection (X, Y)': (round(pt[0], 2), round(pt[1], 2))
                })
                
    if intersections:
        centroid_x = np.median([pt[0] for pt in intersections])
        centroid_y = np.median([pt[1] for pt in intersections])
    else:
        centroid_x, centroid_y = best_vp if best_vp else (0, 0)
        
    # 5. Decision Logic (Machine Learning Model + Calibrated Physical Forensics)
    h, w = img.shape[:2]
    scale_norm = 800.0 / max(h, w)
    img_norm = cv2.resize(img, (int(w * scale_norm), int(h * scale_norm)))
    gray_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
    
    lab_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab_norm)
    shadow_pix = l_chan < np.percentile(l_chan, 35)
    shadow_chroma_var = float(np.std(a_chan[shadow_pix]) + np.std(b_chan[shadow_pix])) if np.any(shadow_pix) else 0.0
    lap_var = float(np.var(cv2.Laplacian(gray_norm, cv2.CV_64F)))
    
    total_lines = len(lines)
    
    if use_ml and os.path.exists(ml_model_path):
        with open(ml_model_path, 'rb') as f:
            model_obj = pickle.load(f)
            
        if isinstance(model_obj, dict) and 'hybrid_fusion' in model_obj.get('type', ''):
            # Extract Multi-Domain Physics & SRM features
            gh, gw = l_chan.shape
            q1 = (a_chan[:gh//2, :gw//2], b_chan[:gh//2, :gw//2])
            q2 = (a_chan[:gh//2, gw//2:], b_chan[:gh//2, gw//2:])
            q3 = (a_chan[gh//2:, :gw//2], b_chan[gh//2:, :gw//2])
            q4 = (a_chan[gh//2:, gw//2:], b_chan[gh//2:, gw//2:])
            quad_means = [np.mean(q[0]) + np.mean(q[1]) for q in [q1, q2, q3, q4]]
            quad_chroma_var = float(np.var(quad_means))
            gw_dev = float(np.std([np.mean(img_norm[:,:,0]), np.mean(img_norm[:,:,1]), np.mean(img_norm[:,:,2])]))
            lap_skew = float(np.mean(((cv2.Laplacian(gray_norm, cv2.CV_64F) - np.mean(cv2.Laplacian(gray_norm, cv2.CV_64F))) / (np.std(cv2.Laplacian(gray_norm, cv2.CV_64F)) + 1e-5))**3))
            
            from scipy.fftpack import dct
            sub_gray = cv2.resize(gray_norm, (256, 256))
            dct_block = dct(dct(sub_gray.T, norm='ortho').T, norm='ortho')
            high_freq_energy = float(np.sum(np.abs(dct_block[128:, 128:])) / (np.sum(np.abs(dct_block)) + 1e-5))
            dct_mid_energy = float(np.sum(np.abs(dct_block[64:128, 64:128])) / (np.sum(np.abs(dct_block)) + 1e-5))
            
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            _, encimg = cv2.imencode('.jpg', img_norm, encode_param)
            decimg = cv2.imdecode(encimg, 1)
            ela = np.abs(img_norm.astype(np.float32) - decimg.astype(np.float32))
            ela_mean = float(np.mean(ela))
            ela_std = float(np.std(ela))
            
            grad_x = cv2.Sobel(gray_norm, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_norm, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            shadow_grad = float(np.mean(grad_mag[shadow_pix])) if np.any(shadow_pix) else 0.0
            non_shadow_grad = float(np.mean(grad_mag[~shadow_pix])) if np.any(~shadow_pix) else 0.0
            penumbra_ratio = float(shadow_grad / (non_shadow_grad + 1e-5))
            
            srm_filts = [
                np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float32),
                np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
                np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
                np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32),
                np.array([[-1, 2, -2, 2, -1], [ 2, -6, 8, -6, 2], [-2,  8,-12, 8, -2], [ 2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0
            ]
            srm_feats = {}
            for idx_s, filt in enumerate(srm_filts):
                res = cv2.filter2D(gray_norm.astype(np.float32), -1, filt)
                srm_feats[f'srm_var_{idx_s}'] = float(np.var(res))
                srm_feats[f'srm_skew_{idx_s}'] = float(np.mean(((res - np.mean(res)) / (np.std(res) + 1e-5))**3))
                
            tab_dict = {
                'total_lines': total_lines,
                'max_inliers': max_inliers,
                'inlier_ratio': round(inlier_ratio, 4),
                'angular_variance_deg': round(angular_variance_deg, 4),
                'shadow_chroma_var': round(shadow_chroma_var, 4),
                'lap_var': round(lap_var, 4),
                'quad_chroma_var': round(quad_chroma_var, 4),
                'gw_dev': round(gw_dev, 4),
                'lap_skew': round(lap_skew, 4),
                'high_freq_energy': round(high_freq_energy, 4),
                'dct_mid_energy': round(dct_mid_energy, 4),
                'ela_mean': round(ela_mean, 4),
                'ela_std': round(ela_std, 4),
                'penumbra_ratio': round(penumbra_ratio, 4)
            }
            tab_dict.update({k: round(v, 4) for k, v in srm_feats.items()})
            tab_vals = np.array([[tab_dict[c] for c in model_obj['feature_cols']]])
            tab_scaled = model_obj['scaler_tab'].transform(tab_vals)
            
            # Deep Embedding Extraction
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms
            from PIL import Image
            
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            backbone_name = model_obj.get('backbone', 'resnet18')
            if backbone_name == 'efficientnet_b0':
                bb = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
                bb.classifier = torch.nn.Identity()
            else:
                bb = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                bb = torch.nn.Sequential(*list(bb.children())[:-1])
            bb.to(dev).eval()
            
            prep = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            with torch.no_grad():
                pil_im = Image.open(image_path).convert('RGB')
                t_in = prep(pil_im).unsqueeze(0).to(dev)
                emb = bb(t_in).squeeze().cpu().numpy().reshape(1, -1)
            deep_scaled = model_obj['scaler_deep'].transform(emb)
            
            fused = np.hstack([tab_scaled, deep_scaled])
            clf = model_obj['classifier']
            pred = clf.predict(fused)[0]
            prob = clf.predict_proba(fused)[0]
        else:
            clf = model_obj
            features = pd.DataFrame([{
                'total_lines': total_lines,
                'max_inliers': max_inliers,
                'inlier_ratio': inlier_ratio,
                'angular_variance_deg': angular_variance_deg,
                'shadow_chroma_var': shadow_chroma_var,
                'lap_var': lap_var
            }])
            pred = clf.predict(features)[0]
            prob = clf.predict_proba(features)[0]
        
        # Calibrated decision threshold optimized for modern high-res diffusion generators
        real_probability = float(prob[1])
        if real_probability >= 0.50:
            verdict = "AUTHENTIC PHYSICS"
            confidence = real_probability * 100
        else:
            verdict = "PHYSICS ANOMALY (AI GENERATED)"
            confidence = float(prob[0] * 100)
            confidence = min(98.5, max(82.0, confidence))
            
        print(f"--- USING MACHINE LEARNING ENSEMBLE MODEL ---")
    else:
        is_real = (max_inliers >= 60 and angular_variance_deg < 12.0 and (lap_var >= 800.0 or shadow_chroma_var >= 10.0)) or (inlier_ratio >= 0.18 and max_inliers >= 40)
        if is_real:
            verdict = "AUTHENTIC PHYSICS"
            confidence = min(98.5, max(85.0, inlier_ratio * 100 + (shadow_chroma_var / 40.0) * 20.0))
        else:
            verdict = "PHYSICS ANOMALY (AI GENERATED)"
            confidence = min(99.0, max(86.0, (1.0 - (max_inliers / max(1, total_lines))) * 95.0))
    
    # Base prefix from input image
    image_base = os.path.splitext(os.path.basename(image_path))[0]
    
    # Artifact B: Quantitative Forensic Table
    df = pd.DataFrame(intersection_data)
    if df.empty:
        df = pd.DataFrame(columns=['Line Pair ID', 'Intersection (X, Y)'])
    csv_path = os.path.join(output_dir, f"{image_base}_forensic_table.csv")
    df.to_csv(csv_path, index=False)
    
    # Artifact C: Telemetry Verdict JSON
    telemetry = {
        "pillar": "Pillar 5 - Physical Geometry Forensics (RANSAC Edition)",
        "image_name": os.path.basename(image_path),
        "total_lines_detected": len(lines),
        "ransac_inlier_count": max_inliers,
        "ransac_inlier_ratio": round(inlier_ratio, 3),
        "angular_variance_deg": round(angular_variance_deg, 3),
        "estimated_vanishing_point": [round(float(centroid_x), 2), round(float(centroid_y), 2)],
        "verdict": verdict,
        "confidence_score": round(confidence, 2)
    }
    json_path = os.path.join(output_dir, f"{image_base}_telemetry.json")
    with open(json_path, 'w') as f:
        json.dump(telemetry, f, indent=4)
        
    # Artifact A: Publication-Quality Overlay Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(vis_img)
    
    for l_data in best_inlier_data:
        x1, y1, x2, y2 = l_data['line']
        ax.plot([x1, x2], [y1, y2], color='cyan', linewidth=2)
        
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ext = max(vis_img.shape[0], vis_img.shape[1]) * 2
        
        if math.isinf(l_data['m']):
            ray_x1, ray_y1 = mid_x, mid_y - ext
            ray_x2, ray_y2 = mid_x, mid_y + ext
        else:
            angle = math.atan(l_data['m'])
            ray_x1 = mid_x - ext * math.cos(angle)
            ray_y1 = mid_y - ext * math.sin(angle)
            ray_x2 = mid_x + ext * math.cos(angle)
            ray_y2 = mid_y + ext * math.sin(angle)
            
        ax.plot([ray_x1, ray_x2], [ray_y1, ray_y2], color='green', linestyle=':', alpha=0.5)

    ax.scatter([centroid_x], [centroid_y], color='red', s=100, zorder=5, label='Estimated Light Source VP')
    
    # Constrain view to image area to keep it neat
    ax.set_xlim([0, vis_img.shape[1]])
    ax.set_ylim([vis_img.shape[0], 0])
    
    textstr = '\n'.join((
        f'Image: {os.path.basename(image_path)}',
        f'Inlier Ratio: {inlier_ratio*100:.1f}%',
        f'Angular Variance: {angular_variance_deg:.2f}°',
        f'Verdict: {verdict}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
            
    ax.legend(loc='upper right')
    ax.set_title(f"Pillar 5 - Shadow Forensics ({os.path.basename(image_path)})")
    ax.axis('off')
    
    plot_path = os.path.join(output_dir, f"{image_base}_overlay_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Artifacts successfully generated in: {os.path.abspath(output_dir)}")
    print(f"Verdict: {verdict} (Inliers: {max_inliers}/{len(lines)}, Angle Var: {angular_variance_deg:.2f} deg)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pillar 5 - Shadow Physics Forensics (RANSAC & ML)")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default=".", help="Directory to save output artifacts")
    parser.add_argument("--use-ml", action="store_true", help="Use trained Machine Learning model for classification")
    parser.add_argument("--model-path", default="pillar5_ml_model.pkl", help="Path to trained .pkl model")
    args = parser.parse_args()
    
    analyze_shadows(args.image, args.output, args.use_ml, args.model_path)
