import pandas as pd
import argparse
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def train_model(csv_path="pillar5_training_dataset.csv", output_model_path="pillar5_ml_model.pkl"):
    if not os.path.exists(csv_path):
        print(f"Dataset {csv_path} not found! Please generate it first.")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset from {csv_path} with {len(df)} samples.")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Selected physics & illumination forensic feature vectors
    feature_cols = [
        'total_lines', 
        'max_inliers', 
        'inlier_ratio', 
        'angular_variance_deg',
        'shadow_chroma_var',
        'lap_var'
    ]
    
    # Extract features X and target y
    X = df[feature_cols]
    y = df['label']
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nTraining Ensemble Physical Forensics Classifier (Random Forest + Gradient Boosting)...")
    
    rf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_split=4, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft'
    )
    
    # Bundle imputer + ensemble together into final pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', ensemble)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n================ MODEL EVALUATION ================")
    print(f"Model Accuracy on Test Set: {acc*100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Fake (AI)", "Authentic"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save trained model pipeline
    with open(output_model_path, 'wb') as f:
        pickle.dump(pipeline, f)
        
    print(f"\nModel successfully saved to: {output_model_path}")
    print("Forensics engine is now ready to classify any uploaded image!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="pillar5_training_dataset.csv", help="Path to training dataset CSV")
    parser.add_argument("--model", default="pillar5_ml_model.pkl", help="Output path for trained model")
    args = parser.parse_args()
    
    train_model(args.csv, args.model)
