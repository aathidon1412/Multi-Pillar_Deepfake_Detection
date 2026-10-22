import cv2
import numpy as np
import pandas as pd
import math
import os
import glob
import random
import pickle
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from xgboost import XGBClassifier

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def extract_efficientnet_embeddings(image_paths, batch_size=32):
    print("Extracting High-Discriminative Multi-Scale Deep Embeddings (EfficientNet-B0)...", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    # Remove classifier to extract 1280-dim feature vector
    model.classifier = torch.nn.Identity()
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
            out = model(batch_stack).cpu().numpy()
            embeddings.append(out)
            if (i // batch_size + 1) % 15 == 0 or i + batch_size >= total:
                print(f"Processed embeddings for {min(i + batch_size, total)}/{total} images...", flush=True)
                
    return np.vstack(embeddings)

def main():
    print("=" * 70, flush=True)
    print("  OPTIMIZING PILLAR 5 ACCURACY ABOVE 90%", flush=True)
    print("=" * 70, flush=True)
    
    # Load multi-domain dataset
    df = pd.read_csv("pillar5_multidomain_dataset.csv")
    print(f"Loaded {len(df)} samples across all 7 AI generators + authentic photos.", flush=True)
    
    feat_cols = [c for c in df.columns if c not in ['img_path', 'label']]
    
    # Extract EfficientNet embeddings
    eff_embs = extract_efficientnet_embeddings(df['img_path'].tolist(), batch_size=32)
    
    # Scale tabular features & deep embeddings
    scaler_tab = RobustScaler()
    tab_imputed = SimpleImputer(strategy='median').fit_transform(df[feat_cols].values)
    tab_scaled = scaler_tab.fit_transform(tab_imputed)
    
    scaler_deep = StandardScaler()
    deep_scaled = scaler_deep.fit_transform(eff_embs)
    
    # Fused 1,304-dimensional multi-domain representation
    X_fused = np.hstack([tab_scaled, deep_scaled])
    y = df['label'].values
    
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y)
    X_train, X_test = X_fused[train_idx], X_fused[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print("\nTraining Optimized Multi-Model Stacking Classifier...", flush=True)
    
    # Base learners
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=3, random_state=42)
    et = ExtraTreesClassifier(n_estimators=300, max_depth=14, min_samples_split=3, random_state=42)
    xgb = XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, random_state=42, eval_metric='logloss')
    lgbm = lgb.LGBMClassifier(n_estimators=350, num_leaves=31, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, random_state=42, verbose=-1)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=400, alpha=0.01, random_state=42)
    
    # Soft Voting Blend of diverse classifiers
    voting_ensemble = VotingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm), ('rf', rf), ('et', et), ('mlp', mlp)],
        voting='soft'
    )
    
    voting_ensemble.fit(X_train, y_train)
    
    y_prob = voting_ensemble.predict_proba(X_test)[:, 1]
    
    # Calculate performance
    y_pred = voting_ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print("\n" + "="*70, flush=True)
    print(f"       >>> FINAL OPTIMIZED TEST ACCURACY: {acc*100:.2f}% <<<", flush=True)
    print("="*70, flush=True)
    print(f"Accuracy : {acc*100:.2f}%", flush=True)
    print(f"Precision: {prec*100:.2f}%", flush=True)
    print(f"Recall   : {rec*100:.2f}%", flush=True)
    print(f"F1-Score : {f1*100:.2f}%", flush=True)
    print(f"ROC-AUC  : {auc*100:.2f}%", flush=True)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Fake (AI)', 'Authentic']), flush=True)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred), flush=True)
    
    # 5-fold cross-validation on full dataset
    cv_scores = cross_val_score(voting_ensemble, X_fused, y, cv=5, scoring='accuracy')
    print(f"\n5-Fold Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)", flush=True)
    
    # Save optimized model bundle
    bundle = {
        'type': 'hybrid_fusion_efficientnet',
        'backbone': 'efficientnet_b0',
        'feature_cols': feat_cols,
        'scaler_tab': scaler_tab,
        'scaler_deep': scaler_deep,
        'classifier': voting_ensemble,
        'accuracy': acc,
        'auc': auc,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open("pillar5_ml_model.pkl", 'wb') as f:
        pickle.dump(bundle, f)
    print("\nOptimized High-Accuracy Model Bundle successfully saved to pillar5_ml_model.pkl!", flush=True)

if __name__ == '__main__':
    main()
