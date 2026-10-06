"""
================================================================================
Pillar 5: Physical Geometry + Hybrid Regularized Ensemble Pipeline
USMFE Multi-Pillar Deepfake Detection Framework
================================================================================

Description:
    This module implements the complete forensic classification pipeline for Pillar 5.
    It fuses 24-dimensional physical & lighting geometry features with 1,280-dimensional
    EfficientNet-B0 embeddings, prunes redundant features, fits a strongly regularized
    multi-model soft-voting ensemble (LightGBM, XGBoost, Random Forest, Extra Trees),
    and evaluates held-out performance.

Key Components:
    1. `reduce_features`: Standardizes inputs using StandardScaler (leakage-free) and
       performs dimensionality reduction or feature selection via:
         - LightGBM / XGBoost SelectFromModel (ranked top-k importance or median)
         - PCA (retaining 95% cumulative explained variance)
       Also extracts and tracks selected feature indices for forensic explainability.
    2. `build_regularized_ensemble`: Instantiates a heavily regularized 4-model
       voting ensemble designed to eliminate overfitting on ~2,000 samples.
    3. `train_and_evaluate_pillar5`: End-to-end orchestration routine.

Usage:
    >>> from pillar5_pipeline import train_and_evaluate_pillar5
    >>> results = train_and_evaluate_pillar5(
    ...     X_train_physics, X_train_effnet, y_train,
    ...     X_test_physics, X_test_effnet, y_test,
    ...     reduction_method="lgbm", n_features=160
    ... )
"""

from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import lightgbm as lgb
import xgboost as xgb


# ==============================================================================
# 1. ROBUST FEATURE REDUCTION & EXPLAINABILITY TRACKING
# ==============================================================================

def reduce_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    method: str = "lgbm",
    n_features: int = 160,
    threshold: Union[str, float] = -np.inf
) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler, BaseEstimator, Optional[np.ndarray]]:
    """
    Standardizes feature vectors and executes dimensionality reduction or feature selection.
    
    Prevents the curse of dimensionality when fusing 24 physical forensic features
    with 1,280 deep visual embeddings on ~2,000 samples.

    Selection Strategy:
        - By default, `threshold=-np.inf` combined with `max_features=n_features`
          guarantees that exactly the top `n_features` most important features are
          selected according to the regularized tree model's feature importance.
        - Alternatively, passing `threshold="median"` selects all features whose
          importance exceeds the median, capped by `max_features`.

    Args:
        X_train (np.ndarray): Training feature matrix of shape (N_samples, 1304).
        y_train (np.ndarray): Binary labels array of shape (N_samples,).
        X_test (np.ndarray, optional): Test feature matrix of shape (N_test, 1304).
        method (str): Reduction strategy: 'lgbm', 'xgb', or 'pca'.
        n_features (int): Number of top informative features to retain for tree methods.
        threshold (Union[str, float]): Importance threshold for SelectFromModel (default: -np.inf).

    Returns:
        Tuple containing:
            - X_train_reduced (np.ndarray): Transformed training features.
            - X_test_reduced (np.ndarray or None): Transformed test features.
            - scaler (StandardScaler): Fitted StandardScaler instance.
            - reducer (SelectFromModel or PCA): Fitted selector/transformer object.
            - selected_indices (np.ndarray or None): 1D array of selected column indices (0-1303).
    """
    method = method.lower()
    if method not in ["lgbm", "xgb", "pca"]:
        raise ValueError(f"Unsupported reduction method '{method}'. Choose from 'lgbm', 'xgb', or 'pca'.")

    print(f"[Pillar 5] Standardizing features (Input shape: {X_train.shape})...")
    
    # Always fit scaler strictly on training set to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    selected_indices: Optional[np.ndarray] = None

    if method == "pca":
        print("[Pillar 5] Fitting PCA to explain 95% cumulative variance...")
        reducer = PCA(n_components=0.95, svd_solver="full", random_state=42)
        X_train_reduced = reducer.fit_transform(X_train_scaled)
        X_test_reduced = reducer.transform(X_test_scaled) if X_test_scaled is not None else None
        print(f"[Pillar 5] PCA retained {reducer.n_components_} components explaining 95% variance.")

    elif method == "lgbm":
        print(f"[Pillar 5] Training regularized LightGBM feature selector (target: {n_features} features)...")
        # Regularized tree estimator for stable feature importances
        base_estimator = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            num_leaves=24,
            learning_rate=0.03,
            min_child_samples=18,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.8,
            reg_lambda=1.2,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        base_estimator.fit(X_train_scaled, y_train)

        # Select top n_features by importance score
        reducer = SelectFromModel(
            estimator=base_estimator,
            max_features=n_features,
            threshold=threshold,
            prefit=True
        )
        X_train_reduced = reducer.transform(X_train_scaled)
        X_test_reduced = reducer.transform(X_test_scaled) if X_test_scaled is not None else None
        selected_indices = reducer.get_support(indices=True)
        print(f"[Pillar 5] LightGBM reduced feature dimensions: {X_train.shape[1]} -> {X_train_reduced.shape[1]}.")

    elif method == "xgb":
        print(f"[Pillar 5] Training regularized XGBoost feature selector (target: {n_features} features)...")
        base_estimator = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.03,
            min_child_weight=4,
            subsample=0.75,
            colsample_bytree=0.65,
            reg_alpha=0.8,
            reg_lambda=1.2,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )
        base_estimator.fit(X_train_scaled, y_train)

        reducer = SelectFromModel(
            estimator=base_estimator,
            max_features=n_features,
            threshold=threshold,
            prefit=True
        )
        X_train_reduced = reducer.transform(X_train_scaled)
        X_test_reduced = reducer.transform(X_test_scaled) if X_test_scaled is not None else None
        selected_indices = reducer.get_support(indices=True)
        print(f"[Pillar 5] XGBoost reduced feature dimensions: {X_train.shape[1]} -> {X_train_reduced.shape[1]}.")

    return X_train_reduced, X_test_reduced, scaler, reducer, selected_indices


# ==============================================================================
# 2. STRONGLY REGULARIZED ENSEMBLE
# ==============================================================================

def build_regularized_ensemble() -> VotingClassifier:
    """
    Builds a strongly regularized soft-voting ensemble across 4 distinct architectures:
    1. LightGBM (Gradient boosted trees with leaf-wise expansion & L1/L2 penalties)
    2. XGBoost (Gradient boosted trees with depth constraints & child weight control)
    3. Random Forest (Bagging with restricted tree depth and leaf constraints)
    4. Extra Trees (Extremely randomized splits for variance reduction)

    Regularization controls:
    - max_depth = 5 or 6 (caps high-order interaction overfitting)
    - min_child_samples / min_samples_leaf >= 16 (prevents memorizing single outlier faces)
    - subsample = 0.75 (row bagging)
    - colsample_bytree / max_features = 0.65 / sqrt (column bagging)
    - reg_alpha = 0.5, reg_lambda = 1.5 (elastic-net style shrinkage on leaf weights)
    - Soft voting with higher weights on boosted models (1.2 vs 1.0 / 0.9)
    """
    # 1. LightGBM
    clf_lgbm = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        num_leaves=24,
        learning_rate=0.03,
        min_child_samples=18,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    # 2. XGBoost
    clf_xgb = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.03,
        min_child_weight=4,
        subsample=0.75,
        colsample_bytree=0.65,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )

    # 3. Random Forest
    clf_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=16,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    # 4. Extra Trees
    clf_et = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=16,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    # Soft-voting ensemble: weighted average of predicted class probabilities
    ensemble = VotingClassifier(
        estimators=[
            ("lgbm", clf_lgbm),
            ("xgb", clf_xgb),
            ("rf", clf_rf),
            ("et", clf_et)
        ],
        voting="soft",
        weights=[1.2, 1.2, 1.0, 0.9],
        n_jobs=-1
    )

    return ensemble


# ==============================================================================
# 3. END-TO-END TRAINING & EVALUATION PIPELINE
# ==============================================================================

def train_and_evaluate_pillar5(
    X_train_physics: np.ndarray,
    X_train_efficientnet: np.ndarray,
    y_train: np.ndarray,
    X_test_physics: np.ndarray,
    X_test_efficientnet: np.ndarray,
    y_test: np.ndarray,
    reduction_method: str = "lgbm",
    n_features: int = 160,
    threshold: Union[str, float] = -np.inf
) -> Dict[str, Any]:
    """
    Executes the complete Pillar 5 forensic workflow:
    1. Concatenates 24-dim physical geometry features with 1,280-dim EfficientNet embeddings (1,304 dims).
    2. Performs robust feature reduction using 'lgbm', 'xgb', or 'pca'.
    3. Fits the strongly regularized 4-model ensemble.
    4. Evaluates predictions on test set (Accuracy, AUC, Classification Report, Confusion Matrix).
    5. Reports explainability metrics regarding retained physics vs deep embedding features.

    Args:
        X_train_physics: (N_train, 24) physical geometry features.
        X_train_efficientnet: (N_train, 1280) deep visual embeddings.
        y_train: (N_train,) ground truth labels (0=Real, 1=Fake).
        X_test_physics: (N_test, 24) physical geometry features.
        X_test_efficientnet: (N_test, 1280) deep visual embeddings.
        y_test: (N_test,) test labels.
        reduction_method: 'lgbm', 'xgb', or 'pca'.
        n_features: Number of selected features (default: 160).
        threshold: Feature importance threshold for SelectFromModel.

    Returns:
        dict: Fitted ensemble, scaler, reducer, selected feature indices, and evaluated metrics.
    """
    print("\n" + "=" * 70)
    print("PILLAR 5: PHYSICAL GEOMETRY + HYBRID ENSEMBLE PIPELINE")
    print("=" * 70)

    # 1. Feature fusion
    print("[1/4] Fusing physical geometry and visual embedding representations...")
    X_train_full = np.hstack([X_train_physics, X_train_efficientnet])
    X_test_full = np.hstack([X_test_physics, X_test_efficientnet])
    print(f" -> Combined train shape: {X_train_full.shape}")
    print(f" -> Combined test shape : {X_test_full.shape}")

    # 2. Dimensionality reduction & explainability
    print(f"\n[2/4] Applying '{reduction_method.upper()}' feature reduction...")
    X_train_reduced, X_test_reduced, scaler, reducer, selected_indices = reduce_features(
        X_train=X_train_full,
        y_train=y_train,
        X_test=X_test_full,
        method=reduction_method,
        n_features=n_features,
        threshold=threshold
    )

    # Analyze how many physical geometry features (indices 0..23) were selected
    if selected_indices is not None:
        physics_selected = [idx for idx in selected_indices if idx < 24]
        deep_selected = [idx for idx in selected_indices if idx >= 24]
        print(f"[Pillar 5 Explainability] Retained {len(physics_selected)}/24 physical geometry features "
              f"and {len(deep_selected)}/1280 deep embeddings.")

    # 3. Ensemble training
    print("\n[3/4] Fitting regularized ensemble (LightGBM, XGBoost, Random Forest, Extra Trees)...")
    ensemble = build_regularized_ensemble()
    ensemble.fit(X_train_reduced, y_train)

    # 4. Evaluation on test set
    print("\n[4/4] Performing inference and evaluation on test split...")
    y_pred = ensemble.predict(X_test_reduced)
    y_probs = ensemble.predict_proba(X_test_reduced)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Real (0)", "Fake (1)"], zero_division=0)

    print("\n" + "=" * 70)
    print("                      TEST SET PERFORMANCE EVALUATION")
    print("=" * 70)
    print(f"Classification Accuracy : {acc * 100:.2f}%")
    print(f"Area Under ROC (ROC-AUC): {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nDetailed Classification Report:")
    print(report)
    print("=" * 70)

    return {
        "ensemble": ensemble,
        "scaler": scaler,
        "reducer": reducer,
        "selected_indices": selected_indices,
        "metrics": {
            "accuracy": acc,
            "auc": auc,
            "confusion_matrix": cm,
            "classification_report": report
        }
    }


# ==============================================================================
# SMOKE TEST / USAGE DEMONSTRATION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING SMOKE TEST: Pillar 5 Pipeline")
    print("=" * 70)

    # Generate synthetic dataset: 100 samples, 24 physics + 1280 deep
    np.random.seed(42)
    n_train, n_test = 80, 20

    X_train_p = np.random.randn(n_train, 24).astype(np.float32)
    X_train_e = np.random.randn(n_train, 1280).astype(np.float32)
    y_tr = np.random.randint(0, 2, size=(n_train,)).astype(np.int32)

    X_test_p = np.random.randn(n_test, 24).astype(np.float32)
    X_test_e = np.random.randn(n_test, 1280).astype(np.float32)
    y_te = np.random.randint(0, 2, size=(n_test,)).astype(np.int32)

    print(f"Synthesized {n_train} train samples, {n_test} test samples.")

    # Run pipeline with LightGBM feature reduction to 50 features for fast test
    results = train_and_evaluate_pillar5(
        X_train_physics=X_train_p,
        X_train_efficientnet=X_train_e,
        y_train=y_tr,
        X_test_physics=X_test_p,
        X_test_efficientnet=X_test_e,
        y_test=y_te,
        reduction_method="lgbm",
        n_features=50
    )

    assert "ensemble" in results
    assert "scaler" in results
    assert "reducer" in results
    assert results["selected_indices"] is not None
    assert len(results["selected_indices"]) == 50

    print("\nSUCCESS: Pillar 5 pipeline passed smoke test verification!")
    print("=" * 70)
