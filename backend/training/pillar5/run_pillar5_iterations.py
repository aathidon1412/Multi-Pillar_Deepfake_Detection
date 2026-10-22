import cv2
import numpy as np
import pandas as pd
import math
import os
import glob
import random
import pickle
import time
from scipy.fftpack import dct
from concurrent.futures import ProcessPoolExecutor
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from xgboost import XGBClassifier

# Set random seeds for deterministic reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def calculate_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0: return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)

# ----------------- SRM (Spatial Rich Model) Filters -----------------
SRM_FILTERS = [
    # 1st order horizontal & vertical
    np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float32),
    np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
    # 2nd order (Laplacian-like)
    np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
    # 3x3 Edge high-pass
    np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32),
    # 5x5 Square SRM
    np.array([[-1, 2, -2, 2, -1],
              [ 2, -6, 8, -6, 2],
              [-2,  8,-12, 8, -2],
              [ 2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0
]

def extract_multi_domain_features(args):
    img_path, label = args
    try:
        img = cv2.imread(str(img_path))
        if img is None: return None
        h, w = img.shape[:2]
        scale = 800.0 / max(h, w)
        img_norm = cv2.resize(img, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
        
        # 1. Baseline Geometry & RANSAC Features
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
        edges = cv2.Canny(thresh, 40, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        
        total_lines = 0
        max_inliers = 0
        inlier_ratio = 0.0
        angular_variance_deg = 90.0
        
        if lines is not None:
            lines = lines.reshape(-1, 4).tolist()
            total_lines = len(lines)
            angles = [math.atan2(l[3]-l[1], l[2]-l[0]) for l in lines]
            angles_mod = np.mod(angles, np.pi)
            median_angle = np.median(angles_mod)
            filtered = [l for l, a in zip(lines, angles_mod) if min(abs(a - median_angle), np.pi - abs(a - median_angle)) < 0.35]
            
            if len(filtered) >= 2:
                best_inlier_lines = []
                for _ in range(min(400, len(filtered)*len(filtered))):
                    l1, l2 = random.sample(filtered, 2)
                    vp = calculate_intersection(l1, l2)
                    if vp is None: continue
                    inl = [l for l in filtered if (abs((l[2]-l[0])*(l[1]-vp[1]) - (l[0]-vp[0])*(l[3]-l[1])) / (math.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2) or 1)) < 50]
                    if len(inl) > max_inliers:
                        max_inliers = len(inl)
                        best_inlier_lines = inl
                inlier_ratio = max_inliers / total_lines if total_lines > 0 else 0
                if best_inlier_lines:
                    inlier_angles = [math.atan2(l[3]-l[1], l[2]-l[0]) for l in best_inlier_lines]
                    ms = np.mean([math.sin(2 * a) for a in inlier_angles])
                    mc = np.mean([math.cos(2 * a) for a in inlier_angles])
                    R = math.sqrt(ms**2 + mc**2)
                    angular_variance_deg = math.degrees((1.0 - R) * np.pi)
                    
        # 2. Illuminant Consistency (CIELAB Color Constancy & Quadrant Variance)
        lab = cv2.cvtColor(img_norm, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        shadow_pix = l_chan < np.percentile(l_chan, 35)
        shadow_chroma_var = float(np.std(a_chan[shadow_pix]) + np.std(b_chan[shadow_pix])) if np.any(shadow_pix) else 0.0
        
        gh, gw = l_chan.shape
        q1 = (a_chan[:gh//2, :gw//2], b_chan[:gh//2, :gw//2])
        q2 = (a_chan[:gh//2, gw//2:], b_chan[:gh//2, gw//2:])
        q3 = (a_chan[gh//2:, :gw//2], b_chan[gh//2:, :gw//2])
        q4 = (a_chan[gh//2:, gw//2:], b_chan[gh//2:, gw//2:])
        quad_means = [np.mean(q[0]) + np.mean(q[1]) for q in [q1, q2, q3, q4]]
        quad_chroma_var = float(np.var(quad_means))
        gw_dev = float(np.std([np.mean(img_norm[:,:,0]), np.mean(img_norm[:,:,1]), np.mean(img_norm[:,:,2])]))
        
        # 3. Laplacian Sensor Residuals
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(np.var(lap))
        lap_skew = float(np.mean(((lap - np.mean(lap)) / (np.std(lap) + 1e-5))**3))
        
        # 4. 2D DCT High-Frequency Energy
        sub_gray = cv2.resize(gray, (256, 256))
        dct_block = dct(dct(sub_gray.T, norm='ortho').T, norm='ortho')
        high_freq_energy = float(np.sum(np.abs(dct_block[128:, 128:])) / (np.sum(np.abs(dct_block)) + 1e-5))
        dct_mid_energy = float(np.sum(np.abs(dct_block[64:128, 64:128])) / (np.sum(np.abs(dct_block)) + 1e-5))
        
        # 5. Error Level Analysis (ELA)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encimg = cv2.imencode('.jpg', img_norm, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        ela = np.abs(img_norm.astype(np.float32) - decimg.astype(np.float32))
        ela_mean = float(np.mean(ela))
        ela_std = float(np.std(ela))
        
        # 6. Shadow Penumbra Edge Gradient Ratio
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        shadow_grad = float(np.mean(grad_mag[shadow_pix])) if np.any(shadow_pix) else 0.0
        non_shadow_grad = float(np.mean(grad_mag[~shadow_pix])) if np.any(~shadow_pix) else 0.0
        penumbra_ratio = float(shadow_grad / (non_shadow_grad + 1e-5))
        
        # 7. Spatial Rich Model (SRM) Noise Residual Statistics
        srm_features = {}
        for idx, filt in enumerate(SRM_FILTERS):
            res = cv2.filter2D(gray.astype(np.float32), -1, filt)
            srm_features[f'srm_var_{idx}'] = float(np.var(res))
            srm_features[f'srm_skew_{idx}'] = float(np.mean(((res - np.mean(res)) / (np.std(res) + 1e-5))**3))
            
        record = {
            'img_path': str(img_path),
            'label': label,
            # Iteration 0 baseline features
            'total_lines': total_lines,
            'max_inliers': max_inliers,
            'inlier_ratio': round(inlier_ratio, 4),
            'angular_variance_deg': round(angular_variance_deg, 4),
            'shadow_chroma_var': round(shadow_chroma_var, 4),
            'lap_var': round(lap_var, 4),
            # Iteration 1 features
            'quad_chroma_var': round(quad_chroma_var, 4),
            'gw_dev': round(gw_dev, 4),
            'lap_skew': round(lap_skew, 4),
            'high_freq_energy': round(high_freq_energy, 4),
            'dct_mid_energy': round(dct_mid_energy, 4),
            'ela_mean': round(ela_mean, 4),
            'ela_std': round(ela_std, 4),
            'penumbra_ratio': round(penumbra_ratio, 4),
        }
        # Add SRM features
        record.update({k: round(v, 4) for k, v in srm_features.items()})
        return record
    except Exception:
        return None

def extract_deep_embeddings(image_paths, batch_size=32):
    print("\nExtracting Deep Visual Embeddings using Pre-Trained Forensic Backbone (ResNet-18)...", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Embedding Extraction Compute Device: {device}", flush=True)
    
    # Load lightweight pre-trained backbone
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Remove the final classification FC layer to get 512-dim feature embedding
    modules = list(model.children())[:-1]
    model = torch.nn.Sequential(*modules)
    model.to(device)
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    embeddings = []
    total = len(image_paths)
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            for p in batch_paths:
                try:
                    pil_img = Image.open(p).convert('RGB')
                    batch_tensors.append(preprocess(pil_img))
                except Exception:
                    batch_tensors.append(torch.zeros(3, 224, 224))
            batch_stack = torch.stack(batch_tensors).to(device)
            out = model(batch_stack)
            out = out.squeeze().cpu().numpy()
            if out.ndim == 1:
                out = out.reshape(1, -1)
            embeddings.append(out)
            if (i // batch_size + 1) % 10 == 0 or i + batch_size >= total:
                print(f"Processed embeddings for {min(i + batch_size, total)}/{total} images...", flush=True)
                
    return np.vstack(embeddings)

def main():
    print("=" * 70, flush=True)
    print("  UNIVERSAL SYNTHETIC MEDIA FORENSICS ENGINE (USMFE) - PILLAR 5", flush=True)
    print("  ACCURACY OPTIMIZATION & MULTI-ITERATION EXPERIMENT BENCHMARK", flush=True)
    print("=" * 70, flush=True)
    
    # 1. Sample balanced dataset
    auth_imgs = [p for p in glob.glob('dataset/authentic/**/*.*', recursive=True) if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.seed(42)
    random.shuffle(auth_imgs)
    auth_sample = auth_imgs[:1000]

    fake_dirs = [d for d in glob.glob('dataset/fake/*') if os.path.isdir(d)]
    fake_sample = []
    per_folder = 1000 // len(fake_dirs)
    for d in fake_dirs:
        imgs = [p for p in glob.glob(d + '/**/*.*', recursive=True) if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.seed(42)
        random.shuffle(imgs)
        fake_sample.extend(imgs[:per_folder])

    tasks = [(p, 1) for p in auth_sample] + [(p, 0) for p in fake_sample]
    print(f"\nExtracted Sample Distribution: {len(auth_sample)} Authentic, {len(fake_sample)} Fake (Total: {len(tasks)}) across 7 AI Generators", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        records = [r for r in ex.map(extract_multi_domain_features, tasks) if r is not None]
    t1 = time.time()
    print(f"Feature extraction completed in {t1-t0:.2f}s for {len(records)} valid samples.", flush=True)
    
    df = pd.DataFrame(records)
    
    # Save extracted dataset
    df.to_csv("pillar5_multidomain_dataset.csv", index=False)
    
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df['label'])
    y_train = df.loc[train_idx, 'label'].values
    y_test = df.loc[test_idx, 'label'].values
    
    reports = {}

    # =========================================================================
    # ITERATION 0: Baseline (6 Heuristic Shadow / Line Features + RF/GB Voting)
    # =========================================================================
    print("\n" + "="*70, flush=True)
    print(">>> ITERATION 0: Baseline Pipeline (6 Handcrafted Geometric Features)", flush=True)
    print("="*70, flush=True)
    iter0_cols = ['total_lines', 'max_inliers', 'inlier_ratio', 'angular_variance_deg', 'shadow_chroma_var', 'lap_var']
    X0_train = df.loc[train_idx, iter0_cols].values
    X0_test = df.loc[test_idx, iter0_cols].values
    
    rf0 = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_split=4, random_state=42)
    gb0 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    pipe0 = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', VotingClassifier(estimators=[('rf', rf0), ('gb', gb0)], voting='soft'))
    ])
    pipe0.fit(X0_train, y_train)
    y_pred0 = pipe0.predict(X0_test)
    y_prob0 = pipe0.predict_proba(X0_test)[:, 1]
    
    acc0 = accuracy_score(y_test, y_pred0)
    prec0 = precision_score(y_test, y_pred0)
    rec0 = recall_score(y_test, y_pred0)
    f1_0 = f1_score(y_test, y_pred0)
    auc0 = roc_auc_score(y_test, y_prob0)
    reports['Iteration 0 (Baseline)'] = {
        'Accuracy': f"{acc0*100:.2f}%", 'Precision': f"{prec0*100:.2f}%", 'Recall': f"{rec0*100:.2f}%", 'F1': f"{f1_0*100:.2f}%", 'AUC': f"{auc0*100:.2f}%"
    }
    print(f"Accuracy: {acc0*100:.2f}% | Precision: {prec0*100:.2f}% | Recall: {rec0*100:.2f}% | F1: {f1_0*100:.2f}% | AUC: {auc0*100:.2f}%", flush=True)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred0), flush=True)

    # =========================================================================
    # ITERATION 1: Multi-Domain Illumination & Spectral Forensics (14 Features)
    # =========================================================================
    print("\n" + "="*70, flush=True)
    print(">>> ITERATION 1: Multi-Domain Light Transport & Frequency Forensics (14 Features)", flush=True)
    print("="*70, flush=True)
    iter1_cols = iter0_cols + ['quad_chroma_var', 'gw_dev', 'lap_skew', 'high_freq_energy', 'dct_mid_energy', 'ela_mean', 'ela_std', 'penumbra_ratio']
    X1_train = df.loc[train_idx, iter1_cols].values
    X1_test = df.loc[test_idx, iter1_cols].values
    
    rf1 = RandomForestClassifier(n_estimators=250, max_depth=10, random_state=42)
    xgb1 = XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.04, subsample=0.85, random_state=42, eval_metric='logloss')
    lgbm1 = lgb.LGBMClassifier(n_estimators=250, num_leaves=31, learning_rate=0.04, subsample=0.85, random_state=42, verbose=-1)
    
    pipe1 = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('clf', VotingClassifier(estimators=[('rf', rf1), ('xgb', xgb1), ('lgbm', lgbm1)], voting='soft'))
    ])
    pipe1.fit(X1_train, y_train)
    y_pred1 = pipe1.predict(X1_test)
    y_prob1 = pipe1.predict_proba(X1_test)[:, 1]
    
    acc1 = accuracy_score(y_test, y_pred1)
    prec1 = precision_score(y_test, y_pred1)
    rec1 = recall_score(y_test, y_pred1)
    f1_1 = f1_score(y_test, y_pred1)
    auc1 = roc_auc_score(y_test, y_prob1)
    reports['Iteration 1 (Multi-Domain Physics)'] = {
        'Accuracy': f"{acc1*100:.2f}%", 'Precision': f"{prec1*100:.2f}%", 'Recall': f"{rec1*100:.2f}%", 'F1': f"{f1_1*100:.2f}%", 'AUC': f"{auc1*100:.2f}%"
    }
    print(f"Accuracy: {acc1*100:.2f}% | Precision: {prec1*100:.2f}% | Recall: {rec1*100:.2f}% | F1: {f1_1*100:.2f}% | AUC: {auc1*100:.2f}%", flush=True)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred1), flush=True)

    # =========================================================================
    # ITERATION 2: Spatial Rich Model (SRM) Noise Steganalysis + Stacking Ensemble (24 Features)
    # =========================================================================
    print("\n" + "="*70, flush=True)
    print(">>> ITERATION 2: Physics + SRM High-Order Noise Residuals & Stacking Classifier", flush=True)
    print("="*70, flush=True)
    iter2_cols = [c for c in df.columns if c not in ['img_path', 'label']]
    X2_train = df.loc[train_idx, iter2_cols].values
    X2_test = df.loc[test_idx, iter2_cols].values
    
    stacking_base2 = [
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, random_state=42, eval_metric='logloss')),
        ('lgbm', lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.03, subsample=0.85, random_state=42, verbose=-1))
    ]
    stacking_clf2 = StackingClassifier(estimators=stacking_base2, final_estimator=LogisticRegression(), cv=5)
    
    pipe2 = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('clf', stacking_clf2)
    ])
    pipe2.fit(X2_train, y_train)
    y_pred2 = pipe2.predict(X2_test)
    y_prob2 = pipe2.predict_proba(X2_test)[:, 1]
    
    acc2 = accuracy_score(y_test, y_pred2)
    prec2 = precision_score(y_test, y_pred2)
    rec2 = recall_score(y_test, y_pred2)
    f1_2 = f1_score(y_test, y_pred2)
    auc2 = roc_auc_score(y_test, y_prob2)
    reports['Iteration 2 (Physics + SRM + Stacking)'] = {
        'Accuracy': f"{acc2*100:.2f}%", 'Precision': f"{prec2*100:.2f}%", 'Recall': f"{rec2*100:.2f}%", 'F1': f"{f1_2*100:.2f}%", 'AUC': f"{auc2*100:.2f}%"
    }
    print(f"Accuracy: {acc2*100:.2f}% | Precision: {prec2*100:.2f}% | Recall: {rec2*100:.2f}% | F1: {f1_2*100:.2f}% | AUC: {auc2*100:.2f}%", flush=True)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred2), flush=True)

    # =========================================================================
    # ITERATION 3: Hybrid Physics & Deep ResNet-18 Embedding Fusion (>90% Target)
    # =========================================================================
    print("\n" + "="*70, flush=True)
    print(">>> ITERATION 3: Hybrid Physics + Deep Visual Embedding Fusion (>90% Target)", flush=True)
    print("="*70, flush=True)
    all_img_paths = df['img_path'].tolist()
    deep_embs = extract_deep_embeddings(all_img_paths, batch_size=32)
    
    # Concatenate Physics Tabular features + 512-dim Deep Embeddings
    scaler_tab = RobustScaler()
    tab_imputed = SimpleImputer(strategy='median').fit_transform(df[iter2_cols].values)
    tab_scaled = scaler_tab.fit_transform(tab_imputed)
    
    scaler_deep = StandardScaler()
    deep_scaled = scaler_deep.fit_transform(deep_embs)
    
    X3_fused = np.hstack([tab_scaled, deep_scaled])
    X3_train = X3_fused[train_idx]
    X3_test = X3_fused[test_idx]
    
    # Hybrid Stacking Meta-Classifier
    xgb3 = XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, random_state=42, eval_metric='logloss')
    lgbm3 = lgb.LGBMClassifier(n_estimators=350, num_leaves=31, learning_rate=0.03, subsample=0.85, random_state=42, verbose=-1)
    rf3 = RandomForestClassifier(n_estimators=350, max_depth=12, random_state=42)
    
    stacking_clf3 = StackingClassifier(
        estimators=[('xgb', xgb3), ('lgbm', lgbm3), ('rf', rf3)],
        final_estimator=LogisticRegression(C=1.0, max_iter=500),
        cv=5
    )
    
    stacking_clf3.fit(X3_train, y_train)
    y_pred3 = stacking_clf3.predict(X3_test)
    y_prob3 = stacking_clf3.predict_proba(X3_test)[:, 1]
    
    acc3 = accuracy_score(y_test, y_pred3)
    prec3 = precision_score(y_test, y_pred3)
    rec3 = recall_score(y_test, y_pred3)
    f1_3 = f1_score(y_test, y_pred3)
    auc3 = roc_auc_score(y_test, y_prob3)
    reports['Iteration 3 (Hybrid Physics + Deep Fusion)'] = {
        'Accuracy': f"{acc3*100:.2f}%", 'Precision': f"{prec3*100:.2f}%", 'Recall': f"{rec3*100:.2f}%", 'F1': f"{f1_3*100:.2f}%", 'AUC': f"{auc3*100:.2f}%"
    }
    print(f"\n================ ITERATION 3 FINAL RESULTS ================", flush=True)
    print(f"Accuracy: {acc3*100:.2f}% | Precision: {prec3*100:.2f}% | Recall: {rec3*100:.2f}% | F1: {f1_3*100:.2f}% | AUC: {auc3*100:.2f}%", flush=True)
    print("\nClassification Report:\n", classification_report(y_test, y_pred3, target_names=['Fake (AI)', 'Authentic']), flush=True)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred3), flush=True)

    # Save the ultimate high-accuracy model bundle to pillar5_ml_model.pkl
    model_bundle = {
        'type': 'hybrid_fusion_stacking',
        'feature_cols': iter2_cols,
        'scaler_tab': scaler_tab,
        'scaler_deep': scaler_deep,
        'classifier': stacking_clf3,
        'accuracy': acc3,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open("pillar5_ml_model.pkl", 'wb') as f:
        pickle.dump(model_bundle, f)
    print("\nHigh-Accuracy Model successfully serialized to pillar5_ml_model.pkl!", flush=True)

    # Print Full Comparative Iteration Summary Table
    print("\n" + "="*80, flush=True)
    print("                      ITERATION COMPARISON REPORT                      ", flush=True)
    print("="*80, flush=True)
    comp_df = pd.DataFrame(reports).T
    print(comp_df.to_string(), flush=True)
    print("="*80, flush=True)

if __name__ == '__main__':
    main()
