#!/usr/bin/python3.12
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def load_model(model_path='shekel_model_expected.keras'):
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
    
    with open('scaler_X_expected.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('scaler_y_expected.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    print("Scalers loaded")
    
    return model, scaler_X, scaler_y


def predict_with_model(X, model_path='shekel_model.keras'):
    """
    Use trained model to make predictions on new data.
    
    Parameters:
    -----------
    X : numpy array or list
        Input features, shape (n_samples, 4) or (4,) for single prediction
    model_path : str
        Path to the saved model
    
    Returns:
    --------
    predictions : numpy array
        Predicted values
    """
    # Load model and scalers
    model, scaler_X, scaler_y = load_model(model_path)
    
    # Convert to numpy array if needed
    X = np.array(X)
    
    # Handle single point vs multiple points
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    # Scale input
    X_scaled = scaler_X.transform(X)
    
    # Predict
    y_pred_scaled = model.predict(X_scaled, verbose=0)
    
    # Inverse transform to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    return y_pred.flatten()


def compare_model_vs_function(test_points, model_path='shekel_model.keras'):
    """
    Compare neural network predictions vs actual Shekel function values.
    
    Parameters:
    -----------
    test_points : list of lists
        Test points to evaluate
    model_path : str
        Path to the saved model
    """
    try:
        from optimization_ex2 import shekel_function
    except ImportError:
        print("Warning: Could not import shekel_function from optimization_ex2")
        return None, None
    
    print("\n" + "=" * 60)
    print("Comparing Model Predictions vs Actual Function")
    print("=" * 60)
    
    # Get predictions from model
    predictions = predict_with_model(test_points, model_path)
    
    # Calculate actual values
    actual_values = np.array([shekel_function(point) for point in test_points])
    
    # Display comparison
    print(f"\n{'Point':<40} {'Actual':<12} {'Predicted':<12} {'Error':<10}")
    print("-" * 80)
    
    for i, point in enumerate(test_points):
        point_str = str([f"{x:.2f}" for x in point])
        actual = actual_values[i]
        pred = predictions[i]
        error = abs(actual - pred)
        print(f"{point_str:<40} {actual:<12.6f} {pred:<12.6f} {error:<10.6f}")
    
    # Calculate statistics
    mae = np.mean(np.abs(actual_values - predictions))
    mse = np.mean((actual_values - predictions)**2)
    rmse = np.sqrt(mse)
    
    print("\n" + "-" * 80)
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Max Error: {np.max(np.abs(actual_values - predictions)):.6f}")
    
    return predictions, actual_values


def optimize_with_model(model_path='shekel_model.keras', bounds=None):
    """
    Optimize the Shekel function using SHGO with the neural network model.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    bounds : list of tuples
        Bounds for each dimension, default [(0, 10)] * 4
    
    Returns:
    --------
    result : scipy.optimize.OptimizeResult
        Optimization result
    """
    from scipy.optimize import shgo
    
    if bounds is None:
        bounds = [(0, 10)] * 4
    
    # Load model and scalers
    model, scaler_X, scaler_y = load_model(model_path)
    
    # Create objective function using the model
    def model_objective(x):
        """Objective function using neural network predictions."""
        x_scaled = scaler_X.transform(x.reshape(1, -1))
        y_pred_scaled = model.predict(x_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        return y_pred[0, 0]
    
    print("\n" + "=" * 60)
    print("Optimizing with Neural Network Model using SHGO")
    print("=" * 60)
    
    # Run SHGO optimization
    result = shgo(model_objective, bounds, n=200, iters=5, sampling_method='simplicial', options={ 'ftol': 1e-3, 'minimize_every_iter': False} )
    
    print(f"\nOptimization Results (Model):")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"\nOptimal solution (x*):")
    print(f"  x = {result.x}")
    print(f"\nOptimal function value (f(x*)):")
    print(f"  f(x*) = {result.fun:.10f}")
    print(f"\nNumber of function evaluations: {result.nfev}")
    print(f"Number of iterations: {result.nit}")
    
    return result


def compare_optimization_results(model_path='shekel_model.keras'):
    """
    Compare SHGO optimization results using actual function vs neural network model.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    """
    try:
        from optimization_ex2 import shekel_function, optimize_shekel
    except ImportError:
        print("Error: Could not import from optimization_ex2")
        return
    
    from scipy.optimize import shgo
    
    bounds = [(0, 10)] * 4
    
    print("\n" + "=" * 80)
    print("COMPARING SHGO OPTIMIZATION: ACTUAL FUNCTION vs NEURAL NETWORK MODEL")
    print("=" * 80)
    
    # 1. Optimize using actual Shekel function
    print("\n[1/2] Running SHGO with ACTUAL Shekel function...")
    print("-" * 80)
    result_actual = shgo(shekel_function, bounds, n=200, iters=5)
    
    # 2. Optimize using neural network model
    print("\n[2/2] Running SHGO with NEURAL NETWORK model...")
    print("-" * 80)
    result_model = optimize_with_model(model_path, bounds)
    
    # 3. Compare results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"\n{'Metric':<30} {'Actual Function':<20} {'NN Model':<20} {'Difference':<15}")
    print("-" * 85)
    
    # Compare optimal x values
    print(f"{'Optimal X:':<30}")
    for i in range(4):
        print(f"  x[{i}]{'':<24} {result_actual.x[i]:<20.6f} {result_model.x[i]:<20.6f} {abs(result_actual.x[i] - result_model.x[i]):<15.6f}")
    
    # Evaluate both solutions with actual function
    actual_at_actual = result_actual.fun
    actual_at_model = shekel_function(result_model.x)
    
    print(f"\n{'Optimal f(x):':<30}")
    print(f"  {'Function result':<27} {actual_at_actual:<20.10f} {result_model.fun:<20.10f} {abs(actual_at_actual - result_model.fun):<15.10f}")
    print(f"  {'Actual Shekel at x*':<27} {actual_at_actual:<20.10f} {actual_at_model:<20.10f} {abs(actual_at_actual - actual_at_model):<15.10f}")
    
    print(f"\n{'Performance:':<30}")
    print(f"  {'Function evaluations':<27} {result_actual.nfev:<20} {result_model.nfev:<20} {result_actual.nfev - result_model.nfev:<15}")
    print(f"  {'Iterations':<27} {result_actual.nit:<20} {result_model.nit:<20} {result_actual.nit - result_model.nit:<15}")
    print(f"  {'Success':<27} {str(result_actual.success):<20} {str(result_model.success):<20}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    x_diff = np.linalg.norm(result_actual.x - result_model.x)
    f_diff = abs(actual_at_actual - actual_at_model)
    
    print(f"Distance between optimal solutions (L2 norm): {x_diff:.6f}")
    print(f"Difference in actual function values: {f_diff:.10f}")
    
    if f_diff < 0.01:
        print("✓ Neural network model found a solution very close to the actual optimum!")
    elif f_diff < 0.1:
        print("~ Neural network model found a reasonably good solution.")
    else:
        print("✗ Neural network model solution differs significantly from actual optimum.")
    
    # Calculate speedup
    if result_actual.nfev > result_model.nfev:
        speedup = result_actual.nfev / result_model.nfev
        print(f"⚡ Speedup: {speedup:.2f}x fewer function evaluations with NN model")
    
    return result_actual, result_model

if __name__ == "__main__":
    # Set random seeds for reproducibility - more comprehensive
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
     
    print("=" * 60)
    
    # Demonstrate using the trained model
    print("\n" + "=" * 60)
    print("Testing trained model on new points:")
    print("=" * 60)
    
    test_points = [
        [4.0, 4.0, 4.0, 4.0],  # Known minimum
        [1.0, 1.0, 1.0, 1.0],
        [8.0, 8.0, 8.0, 8.0],
        [3.5, 4.2, 3.8, 4.1],
        [6.0, 5.5, 6.2, 5.8]
    ]
    
    compare_model_vs_function(test_points)
    # Compare SHGO optimization results
    compare_optimization_results()
