#!/usr/bin/python3.12
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from hashlib import sha256
import logging
import math
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


class ShekelModel(nn.Module):
    """Neural network model for the Shekel function."""
    
    def __init__(self, input_dim=4):
        super(ShekelModel, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.fc2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.fc4 = nn.Linear(32, 16)
        
        self.fc5 = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        
        # Initialize weights using He initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # Set momentum to match Keras default (0.99)
                m.momentum = 0.01  # PyTorch uses 1-momentum, so 0.01 = 0.99 in Keras
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def load_shekel_data(csv_file='shekel_meshgrid_26.csv'):
    """
    Load Shekel function data from CSV file.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    
    Returns:
    --------
    X : numpy array
        Input features (X1, X2, X3, X4)
    y : numpy array
        Target values (Y)
    """
    logging.info(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    logging.info(f"Data shape: {df.shape}")
    logging.info(f"Columns: {df.columns.tolist()}")
    logging.info(f"\nFirst few rows:")
    logging.info(df.head())
    
    # Extract features and target
    X = df[['X1', 'X2', 'X3', 'X4']].values
    y = df['Y'].values
    
    logging.info(f"\nFeatures shape: {X.shape}")
    logging.info(f"Target shape: {y.shape}")
    logging.info(f"Target range: [{y.min():.4f}, {y.max():.4f}]")
    
    return X, y


def cosine_annealing_lr(epoch, max_epochs, initial_lr=0.005, min_lr=1e-6):
    """Cosine annealing learning rate schedule."""
    cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))
    return min_lr + (initial_lr - min_lr) * cosine_decay


def train_model(rootpath=".", csv_file='shekel_meshgrid_26.csv', epochs=100, batch_size=64, test_size=0.2):
    """
    Train a PyTorch neural network model on Shekel function data.
    
    Parameters:
    -----------
    rootpath : str
        Root path for saving outputs
    csv_file : str
        Path to the CSV file
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    test_size : float
        Fraction of data to use for testing
    
    Returns:
    --------
    model : ShekelModel
        Trained model
    scaler_X : StandardScaler
        Scaler for input features
    scaler_y : StandardScaler
        Scaler for target values
    history : dict
        Training history
    results_summary : str
        Summary of results
    """
    # Set device
    device = torch.device('cpu')  # Force CPU as per original code
    logging.info(f"Using device: {device}")
    
    # Load data
    X, y = load_shekel_data(csv_file)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    logging.info(f"\nTrain set size: {X_train.shape[0]}")
    logging.info(f"Test set size: {X_test.shape[0]}")
    
    # Normalize features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled).unsqueeze(1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    logging.info("\nCreating neural network model...")
    model = ShekelModel(input_dim=X_train.shape[1]).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"\nModel Summary:")
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Trainable parameters: {trainable_params}")
    logging.info(model)
    
    # Loss function and optimizer (using NAdam equivalent - AdamW with amsgrad)
    criterion = nn.MSELoss()
    optimizer = optim.NAdam(model.parameters(), lr=0.005)
    
    # Training history
    history = {
        'loss': [],
        'val_loss': [],
        'mae': [],
        'val_mae': [],
        'mse': [],
        'val_mse': [],
        'lr': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience = 40
    patience_counter = 0
    best_model_state = None
    
    logging.info("\nTraining model...")
    logging.info("Initial learning rate: 0.005 (using NAdam optimizer with cosine annealing)")
    
    # Training loop
    for epoch in range(epochs):
        # Update learning rate with cosine annealing
        lr = cosine_annealing_lr(epoch, epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        history['lr'].append(lr)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_mse = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            train_loss += loss.item() * batch_X.size(0)
            train_mae += torch.mean(torch.abs(outputs - batch_y)).item() * batch_X.size(0)
            train_mse += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)
        train_mse /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_mse = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                val_mae += torch.mean(torch.abs(outputs - batch_y)).item() * batch_X.size(0)
                val_mse += loss.item() * batch_X.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_mae /= len(test_loader.dataset)
        val_mse /= len(test_loader.dataset)
        
        # Store history
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        
        # Print progress
        logging.info(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.6f} - mae: {train_mae:.6f} - "
                    f"val_loss: {val_loss:.6f} - val_mae: {val_mae:.6f} - lr: {lr:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"\nEarly stopping triggered at epoch {epoch+1}")
                logging.info(f"Restoring best weights from epoch {epoch+1-patience_counter}")
                break
    
    # Restore best model
    if best_model_state is not None:
        logging.info(f"Restoring best weights from epoch {epoch+1}")
        model.load_state_dict(best_model_state)
    
    # Evaluate model
    model.eval()
    logging.info("\n" + "=" * 60)
    logging.info("Model Evaluation:")
    
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).numpy()
    
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Calculate metrics on original scale
    mse = np.mean((y_test - y_pred.flatten())**2)
    mae = np.mean(np.abs(y_test - y_pred.flatten()))
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y_test - y_pred.flatten())**2) / np.sum((y_test - np.mean(y_test))**2))
    correlation = np.corrcoef(y_test, y_pred.flatten())[0, 1]
    
    metrics_pretty_printed = f"\nMetrics on original scale:\n" + \
                            f"MSE: {mse:.6f}\n" + \
                            f"RMSE: {rmse:.6f}\n" + \
                            f"MAE: {mae:.6f}\n" + \
                            f"R² Score: {r2:.6f}\n" + \
                            f"Correlation Coefficient: {correlation:.6f}"
    
    logging.info(metrics_pretty_printed)
    
    # Test prediction at known minimum
    logging.info("\n" + "=" * 60)
    logging.info("Testing at known minimum [4, 4, 4, 4]:")
    test_point = np.array([[4.0, 4.0, 4.0, 4.0]])
    test_point_scaled = scaler_X.transform(test_point)
    test_point_tensor = torch.FloatTensor(test_point_scaled)
    
    with torch.no_grad():
        pred_scaled = model(test_point_tensor).numpy()
    pred = scaler_y.inverse_transform(pred_scaled)
    
    predicted_pretty_printed = f"Predicted: {pred[0][0]:.4f}"
    logging.info(predicted_pretty_printed)
    logging.info(f"Expected: ~-10.5363")
    
    # Plot training history
    plot_training_history(history, rootpath)
    
    # Plot correlation
    plot_correlation(y_test, y_pred.flatten(), rootpath)
    
    results_summary = metrics_pretty_printed + '\n' + predicted_pretty_printed
    return model, scaler_X, scaler_y, history, results_summary


def plot_correlation(y_true, y_pred, rootpath="."):
    """Plot correlation between actual and predicted values."""
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    
    plt.figure(figsize=(8, 8))
    
    plt.scatter(y_true, y_pred, alpha=0.5, s=10, edgecolors='none')
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_true, p(y_true), 'g-', linewidth=2, alpha=0.7, label=f'Best Fit (y={z[0]:.3f}x+{z[1]:.3f})')
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'Predicted vs Actual\nCorrelation: {correlation:.6f}, R²: {r2:.6f}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    plt.tight_layout()
    plt.savefig(f'{rootpath}/correlation_plot.png', dpi=150)
    logging.info("\nCorrelation plot saved as 'correlation_plot.png'")
    plt.close()


def plot_training_history(history, rootpath="."):
    """Plot training and validation loss over epochs."""
    plt.figure(figsize=(15, 4))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Plot MAE
    plt.subplot(1, 3, 2)
    plt.plot(history['mae'], label='Training MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training and Validation MAE')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(1, 3, 3)
    plt.plot(history['lr'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{rootpath}/training_history.png', dpi=150)
    logging.info("\nTraining history plot saved as 'training_history.png'")
    plt.close()


def save_model(model, scaler_X, scaler_y, rootpath="."):
    """
    Save the scalers and export model to ONNX format.
    
    Parameters:
    -----------
    model : ShekelModel
        Trained model
    scaler_X : StandardScaler
        Scaler for input features
    scaler_y : StandardScaler
        Scaler for target values
    rootpath : str
        Root path for saving
    """
    # Save scalers
    with open(f'{rootpath}/scaler_X_pytorch.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    with open(f'{rootpath}/scaler_y_pytorch.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)
    logging.info("\nScalers saved to scaler_X_pytorch.pkl and scaler_y_pytorch.pkl")
    
    # Export to ONNX
    model.eval()
    dummy_input = torch.randn(1, 4)  # Batch size 1, 4 input features
    onnx_path = f"{rootpath}/shekel_model.pytorch"
    
    # Export with all data embedded in single file
    import io
    f = io.BytesIO()
    torch.onnx.export(
        model,
        dummy_input,
        f,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Write to file
    with open(onnx_path, 'wb') as onnx_file:
        onnx_file.write(f.getvalue())
    
    logging.info(f"ONNX model exported to {onnx_path}")


def main(batch_size: int = 512, rootpath: str = ".") -> str:
    """Main training function."""
    import os
    import random
    
    seed = 42
    
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Set PyTorch to use optimized settings for CPU
    torch.set_num_threads(2)  # Limit threads to avoid overhead on CPU

    logging.info("Random seeds set for reproducibility (seed=42)")
    logging.info("=" * 60)
    
    # Train model
    model, scaler_X, scaler_y, history, results_summary = train_model(
        rootpath,
        csv_file=f'{rootpath}/shekel_meshgrid_26.csv.expected.gz',
        epochs=150,
        batch_size=batch_size,
        test_size=0.2
    )
    
    # Save model (PyTorch and ONNX)
    save_model(model, scaler_X, scaler_y, rootpath)
    
    logging.info("\n" + "=" * 60)
    logging.info("Training complete!")
    
    return sha256(results_summary.encode()).hexdigest()


if __name__ == "__main__":
    import sys
    rootpath = "." if len(sys.argv) < 2 else sys.argv[1]
    batch_size = 512 if len(sys.argv) < 3 else int(sys.argv[2])
    print(main(batch_size, rootpath))
