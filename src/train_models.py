import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, average_precision_score
import xgboost as xgb
import pickle
import os

def train_models():
    print("Loading data...")
    df = pd.read_csv('data/transactions.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Feature Engineering
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    X = df[['amount', 'hour', 'day_of_week']]
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 1. Supervised: XGBoost
    print("Training XGBoost...")
    model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model_xgb.fit(X_train, y_train)
    
    y_pred_xgb = model_xgb.predict(X_test)
    print("XGBoost Performance:")
    print(classification_report(y_test, y_pred_xgb))
    
    # 2. Unsupervised: Isolation Forest
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    iso_forest.fit(X_train)
    
    # Save Models
    os.makedirs('models', exist_ok=True)
    with open('models/model_xgb.pkl', 'wb') as f:
        pickle.dump(model_xgb, f)
        
    with open('models/model_iso.pkl', 'wb') as f:
        pickle.dump(iso_forest, f)
        
    print("Models saved to models/")

if __name__ == "__main__":
    train_models()
