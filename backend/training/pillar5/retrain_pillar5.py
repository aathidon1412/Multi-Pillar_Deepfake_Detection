import os
import glob
import random
import time
import pickle
import math
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fftpack import dct
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler
import lightgbm as lgb
from xgboost import XGBClassifier

# Deterministic seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

FEATURE_COLS = [
    'total_lines', 'max_inliers', 'inlier_ratio', 'angular_variance_deg',
    'shadow_chroma_var', 'lap_var', 'quad_chroma_var', 'gw_dev',
    'lap_skew', 'high_freq_energy', 'dct_mid_energy', 'ela_mean',
    'ela_std', 'penumbra_ratio',
    'srm_var_0', 'srm_skew_0', 'srm_var_1', 'srm_skew_1',
    'srm_var_2', 'srm_skew_2', 'srm_var_3', 'srm_skew_3',
    'srm_var_4', 'srm_skew_4'
]

SRM_FILTERS = [
    np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float32),
    np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
    np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
    np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32),
    np.array([[-1, 2, -2, 2, -1], [ 2, -6, 8, -6, 2], [-2,  8,-12, 8, -2], [ 2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0
]

def extract_single_image_features(item):
    """
    Worker function for parallel tabular feature extraction.
    Takes a tuple (img_path, label) and returns (img_path, label, row_features) or None.
    """
    img_path, label = item
    try:
        img = cv2.imread(img_path)
        if img is None: 
            return None
        h, w = img.shape[:2]
        scale = 800.0 / max(h, w)
        img_norm = cv2.resize(img, (int(w * scale), int(h * scale)))
        gray_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
        
        lab_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab_norm)
        shadow_pix = l_chan < np.percentile(l_chan, 35)
        shadow_chroma_var = float(np.std(a_chan[shadow_pix]) + np.std(b_chan[shadow_pix])) if np.any(shadow_pix) else 0.0
        lap_var = float(np.var(cv2.Laplacian(gray_norm, cv2.CV_64F)))
        
        blurred = cv2.GaussianBlur(gray_norm, (11, 11), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
        edges = cv2.Canny(thresh, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=80, maxLineGap=10)
        total_lines = len(lines) if lines is not None else 0
        
        max_inliers = 0
        angular_variance_deg = 85.0
        if lines is not None and len(lines) >= 2:
            lines_list = lines.reshape(-1, 4).tolist()
            angles = [math.atan2(l[3]-l[1], l[2]-l[0]) for l in lines_list]
            angles_mod = np.mod(angles, np.pi)
            med_ang = np.median(angles_mod)
            fl = [l for l, a in zip(lines_list, angles_mod) if min(abs(a - med_ang), np.pi - abs(a - med_ang)) < 0.35]
            if len(fl) >= 2:
                for _ in range(min(120, len(fl)*len(fl))):
                    l1, l2 = random.sample(fl, 2)
                    den = (l1[0]-l1[2])*(l2[1]-l2[3]) - (l1[1]-l1[3])*(l2[0]-l2[2])
                    if den == 0: continue
                    px = ((l1[0]*l1[3]-l1[1]*l1[2])*(l2[0]-l2[2]) - (l1[0]-l1[2])*(l2[0]*l2[3]-l2[1]*l2[2])) / den
                    py = ((l1[0]*l1[3]-l1[1]*l1[2])*(l2[1]-l2[3]) - (l1[1]-l1[3])*(l2[0]*l2[3]-l2[1]*l2[2])) / den
                    inls = []
                    for line in fl:
                        d_den = math.sqrt((line[2]-line[0])**2 + (line[3]-line[1])**2)
                        d = abs((line[2]-line[0])*(line[1]-py) - (line[0]-px)*(line[3]-line[1])) / d_den if d_den > 0 else 9999
                        if d < 50: inls.append(line)
                    if len(inls) > max_inliers:
                        max_inliers = len(inls)
                        inlier_angles = [math.atan2(l[3]-l[1], l[2]-l[0]) for l in inls]
                        ms = np.mean([math.sin(2*a) for a in inlier_angles])
                        mc = np.mean([math.cos(2*a) for a in inlier_angles])
                        angular_variance_deg = math.degrees((1.0 - math.sqrt(ms**2 + mc**2)) * np.pi)

        inlier_ratio = max_inliers / total_lines if total_lines > 0 else 0.0
        
        gh, gw = l_chan.shape
        q1 = (a_chan[:gh//2, :gw//2], b_chan[:gh//2, :gw//2])
        q2 = (a_chan[:gh//2, gw//2:], b_chan[:gh//2, gw//2:])
        q3 = (a_chan[gh//2:, :gw//2], b_chan[gh//2:, :gw//2])
        q4 = (a_chan[gh//2:, gw//2:], b_chan[gh//2:, gw//2:])
        quad_chroma_var = float(np.var([np.mean(q[0])+np.mean(q[1]) for q in [q1, q2, q3, q4]]))
        gw_dev = float(np.std([np.mean(img_norm[:,:,0]), np.mean(img_norm[:,:,1]), np.mean(img_norm[:,:,2])]))
        lap_arr = cv2.Laplacian(gray_norm, cv2.CV_64F)
        lap_skew = float(np.mean(((lap_arr - np.mean(lap_arr)) / (np.std(lap_arr) + 1e-5))**3))
        
        sub_gray = cv2.resize(gray_norm, (256, 256))
        dct_b = dct(dct(sub_gray.T, norm='ortho').T, norm='ortho')
        high_freq_energy = float(np.sum(np.abs(dct_b[128:, 128:])) / (np.sum(np.abs(dct_b)) + 1e-5))
        dct_mid_energy = float(np.sum(np.abs(dct_b[64:128, 64:128])) / (np.sum(np.abs(dct_b)) + 1e-5))
        
        _, encimg = cv2.imencode('.jpg', img_norm, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
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
        
        srm_feats = {}
        for idx_s, filt in enumerate(SRM_FILTERS):
            res = cv2.filter2D(gray_norm.astype(np.float32), -1, filt)
            srm_feats[f'srm_var_{idx_s}'] = float(np.var(res))
            srm_feats[f'srm_skew_{idx_s}'] = float(np.mean(((res - np.mean(res)) / (np.std(res) + 1e-5))**3))
            
        row = [
            total_lines, max_inliers, round(inlier_ratio, 4), round(angular_variance_deg, 4),
            round(shadow_chroma_var, 4), round(lap_var, 4), round(quad_chroma_var, 4), round(gw_dev, 4),
            round(lap_skew, 4), round(high_freq_energy, 4), round(dct_mid_energy, 4),
            round(ela_mean, 4), round(ela_std, 4), round(penumbra_ratio, 4),
            round(srm_feats['srm_var_0'], 4), round(srm_feats['srm_skew_0'], 4),
            round(srm_feats['srm_var_1'], 4), round(srm_feats['srm_skew_1'], 4),
            round(srm_feats['srm_var_2'], 4), round(srm_feats['srm_skew_2'], 4),
            round(srm_feats['srm_var_3'], 4), round(srm_feats['srm_skew_3'], 4),
            round(srm_feats['srm_var_4'], 4), round(srm_feats['srm_skew_4'], 4)
        ]
        return (img_path, label, row)
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Pillar 5 High-Accuracy Retraining Pipeline")
    parser.add_argument("--samples", type=int, default=8000, help="Total training samples (balanced 50% Real / 50% AI across 7 generators)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Parallel CPU workers for feature extraction")
    args = parser.parse_args()

    total_samples = args.samples
    num_auth = total_samples // 2
    num_fake = total_samples // 2

    print("=" * 75, flush=True)
    print(f"   USMFE PILLAR 5 HIGH-SCALE RETRAINING PIPELINE (Target: {total_samples} samples)", flush=True)
    print("=" * 75, flush=True)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    auth_paths = [f for f in glob.glob(os.path.join(base_dir, 'dataset', 'authentic', '**', '*.*'), recursive=True) if not os.path.isdir(f)]
    
    # Check for Kaggle train_data & train.csv integration
    kaggle_csv = os.path.join(base_dir, 'train.csv')
    kaggle_train_dir = os.path.join(base_dir, 'dataset', 'train_data')
    kaggle_auth = []
    kaggle_fake = []
    if os.path.exists(kaggle_csv) and os.path.exists(kaggle_train_dir):
        df_k = pd.read_csv(kaggle_csv)
        for _, row in df_k.iterrows():
            img_p = os.path.join(kaggle_train_dir, os.path.basename(row['file_name']))
            if os.path.exists(img_p):
                if row['label'] == 1:
                    kaggle_auth.append(img_p)
                else:
                    kaggle_fake.append(img_p)
        print(f"Kaggle Dataset detected: {len(kaggle_auth)} Authentic + {len(kaggle_fake)} AI images", flush=True)

    fake_subdirs = [d for d in glob.glob(os.path.join(base_dir, 'dataset', 'fake', '*')) if os.path.isdir(d)]
    
    total_auth_pool = auth_paths + kaggle_auth
    random.shuffle(total_auth_pool)
    selected_auth = total_auth_pool[:min(num_auth, len(total_auth_pool))]
    
    print(f"Total Authentic pool in storage: {len(total_auth_pool)}", flush=True)
    print(f"Generative domains ({len(fake_subdirs)}): {[os.path.basename(d) for d in fake_subdirs]}", flush=True)
    
    # Sample balanced Fake across all generator subdirectories + Kaggle AI pool
    selected_fake = []
    if kaggle_fake:
        random.shuffle(kaggle_fake)
        k_count = min(len(kaggle_fake), num_fake // 2)
        selected_fake.extend(kaggle_fake[:k_count])
        print(f" -> Sampled {k_count} from 'Kaggle Modern AI Pool'", flush=True)
        rem_fake = num_fake - k_count
    else:
        rem_fake = num_fake

    per_gen = rem_fake // max(1, len(fake_subdirs))
    for sd in fake_subdirs:
        gen_files = [f for f in glob.glob(os.path.join(sd, '**', '*.*'), recursive=True) if not os.path.isdir(f)]
        random.shuffle(gen_files)
        picked = gen_files[:min(len(gen_files), per_gen)]
        selected_fake.extend(picked)
        print(f" -> Sampled {len(picked)} from '{os.path.basename(sd)}'", flush=True)
        
    print(f"\nFinal Selected Pool: {len(selected_auth)} Authentic + {len(selected_fake)} AI-Generated = {len(selected_auth)+len(selected_fake)} Total", flush=True)
    
    all_records = [(p, 1) for p in selected_auth] + [(p, 0) for p in selected_fake]
    random.shuffle(all_records)
    
    # 1. Parallel Tabular Feature Extraction
    print(f"\n[1/3] Extracting 24 Physics & Steganalysis Features using {args.workers} Parallel CPU Workers...", flush=True)
    t0 = time.time()
    tab_data = []
    labels = []
    valid_paths = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract_single_image_features, item) for item in all_records]
        done_count = 0
        for fut in as_completed(futures):
            res = fut.result()
            done_count += 1
            if res is not None:
                p, label, feat = res
                tab_data.append(feat)
                labels.append(label)
                valid_paths.append(p)
            if done_count % 500 == 0 or done_count == len(all_records):
                elapsed = time.time() - t0
                speed = done_count / max(1, elapsed)
                print(f" -> Extracted {done_count}/{len(all_records)} images ({speed:.1f} img/sec)...", flush=True)
                
    X_tab = np.array(tab_data)
    y = np.array(labels)
    print(f"Tabular extraction completed in {time.time()-t0:.1f}s. Valid samples: {len(valid_paths)}", flush=True)
    
    # 2. Extract Multi-Scale Deep Visual Embeddings (EfficientNet-B0)
    print("\n[2/3] Extracting 1,280-dim Deep Visual Embeddings (EfficientNet-B0)...", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Compute Device: {device}", flush=True)
    
    bb = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    bb.classifier = torch.nn.Identity()
    bb.to(device).eval()
    
    prep = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    embeddings = []
    batch_size = 64
    t1 = time.time()
    with torch.no_grad():
        for i in range(0, len(valid_paths), batch_size):
            batch_paths = valid_paths[i:i+batch_size]
            tensors = []
            for p in batch_paths:
                try:
                    tensors.append(prep(Image.open(p).convert('RGB')))
                except Exception:
                    tensors.append(torch.zeros(3, 224, 224))
            batch_t = torch.stack(tensors).to(device)
            out = bb(batch_t).cpu().numpy()
            embeddings.append(out)
            if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(valid_paths):
                print(f" -> Deep features: {min(i + batch_size, len(valid_paths))}/{len(valid_paths)} images processed...", flush=True)
                
    X_deep = np.vstack(embeddings)
    print(f"Deep feature extraction completed in {time.time()-t1:.1f}s. Embedding shape: {X_deep.shape}", flush=True)
    
    # 3. Scale, Fuse, and Train Multi-Model Soft Voting Ensemble
    print("\n[3/3] Scaling, Fusing (1,304-dim), and Training 4-Model Soft Voting Ensemble...", flush=True)
    scaler_tab = RobustScaler()
    tab_scaled = scaler_tab.fit_transform(X_tab)
    
    scaler_deep = StandardScaler()
    deep_scaled = scaler_deep.fit_transform(X_deep)
    
    X_fused = np.hstack([tab_scaled, deep_scaled])
    
    X_train, X_test, y_train, y_test = train_test_split(X_fused, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training XGBoost Classifier...", flush=True)
    xgb = XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, random_state=42, eval_metric='logloss', n_jobs=-1)
    
    print("Training LightGBM Classifier...", flush=True)
    lgbm = lgb.LGBMClassifier(n_estimators=350, num_leaves=31, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, verbose=-1)
    
    print("Training Random Forest Classifier...", flush=True)
    rf = RandomForestClassifier(n_estimators=250, max_depth=14, min_samples_split=3, random_state=42, n_jobs=-1)
    
    print("Training Extra Trees Classifier...", flush=True)
    et = ExtraTreesClassifier(n_estimators=250, max_depth=16, min_samples_split=3, random_state=42, n_jobs=-1)
    
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm), ('rf', rf), ('et', et)],
        voting='soft',
        n_jobs=-1
    )
    
    ensemble.fit(X_train, y_train)
    
    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print("\n" + "=" * 75)
    print(f"       >>> MODEL ACCURACY: {acc*100:.2f}% | ROC-AUC: {auc*100:.2f}% <<<")
    print("=" * 75)
    print(classification_report(y_test, y_pred, target_names=['Fake (AI)', 'Authentic']))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Save Authoritative Model Bundle
    bundle = {
        'type': 'hybrid_fusion_efficientnet',
        'backbone': 'efficientnet_b0',
        'feature_cols': FEATURE_COLS,
        'scaler_tab': scaler_tab,
        'scaler_deep': scaler_deep,
        'classifier': ensemble,
        'accuracy': acc,
        'auc': auc,
        'total_trained_samples': len(valid_paths),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    out_model_path = os.path.join(base_dir, "pillar5_ml_model.pkl")
    with open(out_model_path, 'wb') as f:
        pickle.dump(bundle, f)
        
    print(f"\n===========================================================================")
    print(f"SUCCESS: Authoritative Pillar 5 Model saved to: {out_model_path}")
    print(f"Engine is calibrated and ready to serve all AI generator detection tasks!")
    print(f"===========================================================================\n")

if __name__ == '__main__':
    main()

