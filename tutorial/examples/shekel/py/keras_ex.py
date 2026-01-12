#!/usr/bin/python3.12
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def load_shekel_data(csv_file='shekel_meshgrid_26.csv.expected.gz'):
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
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Extract features and target
    X = df[['X1', 'X2', 'X3', 'X4']].values
    y = df['Y'].values
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")
    
    return X, y


def create_shekel_model(input_dim=4):
    """
    Create a neural network model for the Shekel function.
    
    A smaller network is sufficient for smooth mathematical functions.
    
    Parameters:
    -----------
    input_dim : int
        Number of input features (default: 4)
    
    Returns:
    --------
    model : keras.Model
        Compiled neural network model
    """
    # Use He initialization for better convergence with ReLU
    initializer = keras.initializers.HeNormal()
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_dim,)),
        
        # Hidden layers with proper initialization
        layers.Dense(32, activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(),
        
        layers.Dense(64, activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(),
        
        layers.Dense(32, activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(),
        
        layers.Dense(16, activation='relu', kernel_initializer=initializer),
        
        # Output layer
        layers.Dense(1, kernel_initializer=initializer)
    ])
    
    # Use Nadam optimizer (Adam + Nesterov momentum) for faster convergence
    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=0.005),  # Higher learning rate with Nadam
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


def train_model(csv_file='shekel_meshgrid_26.csv', epochs=100, batch_size=64, test_size=0.2):
    """
    Train a neural network model on Shekel function data.
    
    Parameters:
    -----------
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
    model : keras.Model
        Trained model
    scaler_X : StandardScaler
        Scaler for input features
    scaler_y : StandardScaler
        Scaler for target values
    history : keras.callbacks.History
        Training history
    """
    # Load data
    X, y = load_shekel_data(csv_file)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Normalize features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Create model
    print("\nCreating neural network model...")
    model = create_shekel_model(input_dim=X_train.shape[1])
    print(model.summary())
    
    # Early stopping callback with adjusted patience
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=40,
        restore_best_weights=True,
        verbose=1
    )
    
    # Cosine annealing learning rate schedule - smooth decay from start
    def cosine_decay_schedule(epoch, lr):
        """Cosine annealing schedule for smooth learning rate decay."""
        import math
        max_epochs = epochs
        initial_lr = 0.005
        min_lr = 1e-6
        # Cosine decay
        cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))
        decayed_lr = min_lr + (initial_lr - min_lr) * cosine_decay
        return decayed_lr
    
    lr_scheduler = keras.callbacks.LearningRateScheduler(cosine_decay_schedule, verbose=0)
    
    # Custom callback to log learning rate
    class LearningRateLogger(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            logs['lr'] = float(self.model.optimizer.learning_rate)
    
    lr_logger = LearningRateLogger()
    
    # Optional: Reduce LR on plateau as backup (less aggressive now)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train model
    print("\nTraining model...")
    print("Initial learning rate: 0.005 (using Nadam optimizer with cosine annealing)")
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_scaled, y_test_scaled),
        callbacks=[lr_scheduler, lr_logger, early_stopping, reduce_lr],
        verbose=2  # Show one line per epoch
    )
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("Model Evaluation:")
    results = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
    test_loss = results[0]
    test_mae = results[1]
    test_mse = results[2]
    print(f"Test Loss (MSE): {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    
    # Make predictions
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Calculate metrics on original scale
    mse = np.mean((y_test - y_pred.flatten())**2)
    mae = np.mean(np.abs(y_test - y_pred.flatten()))
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y_test - y_pred.flatten())**2) / np.sum((y_test - np.mean(y_test))**2))
    correlation = np.corrcoef(y_test, y_pred.flatten())[0, 1]
    
    print(f"\nMetrics on original scale:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R² Score: {r2:.6f}")
    print(f"Correlation Coefficient: {correlation:.6f}")
    
    # Test prediction at known minimum
    print("\n" + "=" * 60)
    print("Testing at known minimum [4, 4, 4, 4]:")
    test_point = np.array([[4.0, 4.0, 4.0, 4.0]])
    test_point_scaled = scaler_X.transform(test_point)
    pred_scaled = model.predict(test_point_scaled, verbose=0)
    pred = scaler_y.inverse_transform(pred_scaled)
    print(f"Predicted: {pred[0][0]:.6f}")
    print(f"Expected: ~-10.5363")
    
    # Plot training history
    plot_training_history(history)
    
    # Plot correlation
    plot_correlation(y_test, y_pred.flatten())
    
    return model, scaler_X, scaler_y, history


def plot_correlation(y_true, y_pred):
    """
    Plot correlation between actual and predicted values.
    
    Parameters:
    -----------
    y_true : numpy array
        Actual values
    y_pred : numpy array
        Predicted values
    """
    # Calculate correlation coefficient and R²
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    
    plt.figure(figsize=(8, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, s=10, edgecolors='none')
    
    # Perfect prediction line (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Best fit line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_true, p(y_true), 'g-', linewidth=2, alpha=0.7, label=f'Best Fit (y={z[0]:.3f}x+{z[1]:.3f})')
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'Predicted vs Actual\nCorrelation: {correlation:.6f}, R²: {r2:.6f}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Set same limits for both axes
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    plt.tight_layout()
    plt.savefig('correlation_plot.png', dpi=150)
    print("\nCorrelation plot saved as 'correlation_plot.png'")
    plt.close()


def plot_training_history(history):
    """
    Plot training and validation loss over epochs.
    
    Parameters:
    -----------
    history : keras.callbacks.History
        Training history
    """
    plt.figure(figsize=(15, 4))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Log scale to see details
    
    # Plot MAE
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training and Validation MAE')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(1, 3, 3)
    if 'lr' in history.history:
        plt.plot(history.history['lr'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("\nTraining history plot saved as 'training_history.png'")
    plt.close()


def save_model(model, scaler_X, scaler_y, model_path='shekel_model.keras'):
    """
    Save the trained model and scalers.
    
    Parameters:
    -----------
    model : keras.Model
        Trained model
    scaler_X : StandardScaler
        Scaler for input features
    scaler_y : StandardScaler
        Scaler for target values
    model_path : str
        Path to save the model
    """
    # Save model
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save scalers
    import pickle
    with open('scaler_X.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    with open('scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)
    print("Scalers saved to scaler_X.pkl and scaler_y.pkl")


def load_model(model_path='shekel_model.keras'):
    """
    Load a saved model and scalers.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    
    Returns:
    --------
    model : keras.Model
        Loaded model
    scaler_X : StandardScaler
        Scaler for input features
    scaler_y : StandardScaler
        Scaler for target values
    """
    import pickle
    
    model = keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    with open('scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    print("Scalers loaded")
    
    return model, scaler_X, scaler_y


if __name__ == "__main__":
    # Set random seeds for reproducibility - more comprehensive
    import os
    import random
    
    seed = 42
    
    # Python random
    random.seed(seed)
    
    # Numpy random
    np.random.seed(seed)
    
    # TensorFlow random
    tf.random.set_seed(seed)
    
    # Set Python hash seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    
    # Configure TensorFlow for deterministic operations
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    # For GPU determinism (if using GPU)
    #tf.config.experimental.enable_op_determinism()
    
    print("Random seeds set for reproducibility (seed=42)")
    print("=" * 60)
    
    # Train model with optimized parameters for faster convergence
    model, scaler_X, scaler_y, history = train_model(
        csv_file='shekel_meshgrid_26.csv.expected.gz',
        epochs=150,  # Reduced from 200, early stopping will handle it
        batch_size=512,  # Larger batch for faster, more stable training
        test_size=0.2
    )
    
    # Save model
    save_model(model, scaler_X, scaler_y)
    
    print("\n" + "=" * 60)
    print("Training complete!")
