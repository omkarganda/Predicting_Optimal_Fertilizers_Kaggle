import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp  # For mixed precision training
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import top_k_accuracy_score
from sklearn.linear_model import LogisticRegression
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # For advanced logging
import joblib
from tqdm import tqdm
import time
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Enhanced Neural Network with Residual Connections
class FertilizerNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if input_size != 512:
            self.shortcut = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.BatchNorm1d(512)
            )
        
    def forward(self, x):
        identity = x
        
        # Block 1
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        identity = self.shortcut(identity)
        out += identity
        
        # Block 2
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Block 3
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Output
        return self.fc4(out)

# 2. Optimized Training Function with Progress Tracking
def train_model(model, train_loader, val_loader, device, epochs=100, patience=5, 
                model_idx=0, fold_idx=0, writer=None):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    scaler = amp.GradScaler()  # For mixed precision training
    
    best_val_loss = float('inf')
    best_weights = None
    no_improve = 0
    
    # Progress bar
    epoch_pbar = tqdm(range(epochs), desc=f"Model {model_idx+1} Fold {fold_idx+1}")
    
    for epoch in epoch_pbar:
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        batch_pbar = tqdm(train_loader, desc=f"Batch", leave=False)
        for inputs, labels in batch_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Mixed precision context
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Backpropagation
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Calculate metrics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update batch progress
            batch_pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100.*correct/total
            })
        
        train_loss /= len(train_loader.dataset)
        train_acc = 100.*correct/total
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                with amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                
                probs = torch.softmax(outputs.float(), dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_probs = np.vstack(all_probs)
        val_labels = np.concatenate(all_labels)
        
        # Calculate MAP@3
        map3 = top_k_accuracy_score(val_labels, val_probs, k=3) * 100
        
        # TensorBoard logging
        if writer:
            writer.add_scalar(f'model{model_idx}/fold{fold_idx}/train_loss', train_loss, epoch)
            writer.add_scalar(f'model{model_idx}/fold{fold_idx}/train_acc', train_acc, epoch)
            writer.add_scalar(f'model{model_idx}/fold{fold_idx}/val_loss', val_loss, epoch)
            writer.add_scalar(f'model{model_idx}/fold{fold_idx}/val_map3', map3, epoch)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Update epoch progress
        epoch_pbar.set_postfix({
            'train_loss': f"{train_loss:.4f}",
            'val_loss': f"{val_loss:.4f}",
            'val_map3': f"{map3:.2f}%"
        })
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                epoch_pbar.set_postfix({
                    'train_loss': f"{train_loss:.4f}",
                    'val_loss': f"{val_loss:.4f}",
                    'val_map3': f"{map3:.2f}%",
                    'status': 'Early stopping'
                })
                break
    
    # Load best weights
    model.load_state_dict(best_weights)
    return model, val_probs

# 3. GPU-Optimized Main Pipeline
def train_and_predict(X_train_tensor, y_train_tensor, X_test_tensor, test_ids, 
                      le, n_models=3, n_splits=3, device='cuda'):
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir='logs')
    
    num_classes = len(le.classes_)
    input_size = X_train_tensor.shape[1]
    
    # Initialize storage
    oof_preds = np.zeros((len(X_train_tensor), num_classes))
    test_preds = np.zeros((len(X_test_tensor), num_classes))
    meta_features_train = np.zeros((len(X_train_tensor), n_models * num_classes))
    meta_features_test = np.zeros((len(X_test_tensor), n_models * num_classes))
    
    # K-Fold setup
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Convert tensors to numpy for sklearn
    y_train_np = y_train_tensor.numpy()
    
    # Train ensemble of models
    for model_idx in range(n_models):
        model_start = time.time()
        print(f"\n{'='*50}")
        print(f"Training Model {model_idx+1}/{n_models}")
        print(f"{'='*50}")
        
        model_test_preds = np.zeros((len(X_test_tensor), num_classes))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_tensor, y_train_np)):
            fold_start = time.time()
            print(f"\n  Fold {fold+1}/{n_splits} - Model {model_idx+1}")
            
            # Create data loaders with pinned memory
            train_dataset = TensorDataset(
                X_train_tensor[train_idx], 
                y_train_tensor[train_idx]
            )
            val_dataset = TensorDataset(
                X_train_tensor[val_idx], 
                y_train_tensor[val_idx]
            )
            test_dataset = TensorDataset(X_test_tensor)
            
            # Use pinned memory and multiple workers
            train_loader = DataLoader(
                train_dataset, 
                batch_size=1024, 
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=1024, 
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=1024, 
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            
            # Initialize and train model
            model = FertilizerNN(input_size, num_classes)
            trained_model, val_probs = train_model(
                model, train_loader, val_loader, device,
                epochs=50, 
                patience=5,
                model_idx=model_idx,
                fold_idx=fold,
                writer=writer
            )
            
            # Store OOF predictions
            oof_preds[val_idx] += val_probs / n_models
            
            # Get test predictions
            trained_model.eval()
            test_probs = []
            
            with torch.no_grad():
                for inputs in test_loader:
                    inputs = inputs[0].to(device)
                    outputs = trained_model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    test_probs.append(probs.cpu().numpy())
            
            test_fold_probs = np.vstack(test_probs)
            model_test_preds += test_fold_probs / n_splits
            
            # Free GPU memory
            del trained_model, model
            torch.cuda.empty_cache()
            
            fold_time = time.time() - fold_start
            print(f"  Fold {fold+1} completed in {fold_time:.2f} seconds")
        
        # Store model's test predictions
        test_preds += model_test_preds / n_models
        
        # Add to meta features for stacking
        start_idx = model_idx * num_classes
        end_idx = (model_idx + 1) * num_classes
        meta_features_train[:, start_idx:end_idx] = oof_preds
        meta_features_test[:, start_idx:end_idx] = model_test_preds
        
        model_time = time.time() - model_start
        print(f"\nModel {model_idx+1} completed in {model_time:.2f} seconds")
    
    # Stacking Ensemble
    print("\nTraining Stacking Classifier...")
    stacker = LogisticRegression(
        multi_class='multinomial',
        solver='saga',
        max_iter=2000,
        penalty='elasticnet',
        l1_ratio=0.5,
        C=0.1,
        n_jobs=-1,
        random_state=42
    )
    stacker.fit(meta_features_train, y_train_np)
    
    # Final predictions
    stack_test_preds = stacker.predict_proba(meta_features_test)
    top3_preds = np.argsort(-stack_test_preds, axis=1)[:, :3]
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_ids,
        'Fertilizer Name': [
            ' '.join(le.inverse_transform([top3_preds[i, 0], top3_preds[i, 1], top3_preds[i, 2]])) 
            for i in range(len(top3_preds))
        ]
    })
    
    # Close TensorBoard writer
    writer.close()
    
    return submission

# 4. Execution Wrapper
# def main():
#     # Detect device
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")
    
#     if device == 'cuda':
#         print(f"GPU: {torch.cuda.get_device_name(0)}")
#         print(f"CUDA version: {torch.version.cuda}")
    
#     # Load your preprocessed data
#     # Replace these with your actual data loading code
#     X_train_tensor = torch.load('X_train_tensor.pt')
#     y_train_tensor = torch.load('y_train_tensor.pt')
#     X_test_tensor = torch.load('X_test_tensor.pt')
#     test_ids = np.load('test_ids.npy')
#     le = joblib.load('label_encoder.pkl')
    
#     # Start training
#     start_time = time.time()
#     submission, stacker, oof_preds, test_preds = train_and_predict(
#         X_train_tensor, y_train_tensor, X_test_tensor, test_ids, le,
#         n_models=3, n_splits=3, device=device
#     )
    
#     # Save outputs
#     submission.to_csv('submission.csv', index=False)
#     joblib.dump(stacker, 'stacking_ensemble.pkl')
#     np.save('oof_predictions.npy', oof_preds)
#     np.save('test_predictions.npy', test_preds)
    
#     # Calculate final OOF MAP@3
#     y_train_np = y_train_tensor.numpy()
#     oof_map3 = top_k_accuracy_score(y_train_np, oof_preds, k=3) * 100
#     print(f"\nFinal OOF MAP@3: {oof_map3:.2f}%")
    
#     total_time = time.time() - start_time
#     print(f"\nTraining complete in {total_time/3600:.2f} hours")
#     print(f"Submission file created: submission.csv")
#     print(f"Sample submission:\n{submission.head()}")

# if __name__ == '__main__':
#     main()