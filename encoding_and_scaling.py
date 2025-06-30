import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch

def auto_preprocess_data(train_df, test_df, target_col='Fertilizer Name'):
    """
    Automatically detects and processes numeric/categorical columns
    Returns:
    - X_train, y_train (processed training data)
    - X_test (processed test data)
    - preprocessor (fitted preprocessing pipeline)
    - label_encoder (fitted target encoder)
    """
    # Make copies to avoid modifying original data
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Automatic column type detection
    numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target from features if present
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    # Separate features and target
    y_train = train_df[target_col]
    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col], errors='ignore')
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])
    
    # Encode target
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, label_encoder