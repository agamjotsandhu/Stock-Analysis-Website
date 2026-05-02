import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def build_model():
    # 1. Load data
    print("Loading data...")
    df = pd.read_csv('master_etf_data.csv')
    
    # 2. Filter for SPY
    df_spy = df[df['Ticker'] == 'SPY'].copy()
    df_spy['Date'] = pd.to_datetime(df_spy['Date'])
    df_spy = df_spy.sort_values('Date')
    
    # 3. Create Target Variable (No Data Leakage)
    # We want to predict if Close > Open the FOLLOWING day.
    # Target = 1 if tomorrow's Close > tomorrow's Open, 0 otherwise.
    # We shift this back so that today's row contains the target for tomorrow's prediction.
    df_spy['target'] = (df_spy['Close'].shift(-1) > df_spy['Open'].shift(-1)).astype(int)
    
    # 4. Feature Selection
    # We use today's data to predict tomorrow's outcome.
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'RSI_14']
    
    # Drop rows with NaNs (first 50 rows due to SMA/RSI and last row due to shift)
    df_spy = df_spy.dropna(subset=features + ['target'])
    
    X = df_spy[features]
    y = df_spy['target']
    
    # 5. Split data (80% train, 20% test)
    # Note: For time series, usually we don't shuffle, but the prompt says 80/20 split. 
    # I'll use shuffle=False to respect the temporal nature of stock data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    
    # 6. Train XGBoost Model
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # 7. Evaluate Model
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 8. Feature Importance
    print("\nSaving feature importance chart...")
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.title("Feature Importances (SPY Prediction)")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance chart saved as 'feature_importance.png'.")

if __name__ == "__main__":
    build_model()
