#!/usr/bin/python3.12
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sys import argv
from hashlib import sha256
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

def load_model(rootpath = ".", model_path='shekel_model_expected.keras'):
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
    
    model = keras.models.load_model(rootpath + "/" + model_path)
    logging.info(f"Model loaded from {model_path}")
    
    with open(rootpath + '/scaler_X_expected.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open(rootpath + '/scaler_y_expected.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    logging.info("Scalers loaded")
    
    return model, scaler_X, scaler_y


def predict_with_model(X, rootpath = ".", model_path='shekel_model_expected.keras'):
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
    model, scaler_X, scaler_y = load_model(rootpath, model_path)
    
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


def compare_model_vs_function(test_points, rootpath=".", model_path='shekel_model_expected.keras'):
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
        logging.info("Warning: Could not import shekel_function from optimization_ex2")
        return None, None
    
    logging.info("\n" + "=" * 60)
    logging.info("Comparing Model Predictions vs Actual Function")
    logging.info("=" * 60)
    
    # Get predictions from model
    predictions = predict_with_model(test_points, rootpath, model_path)
    
    # Calculate actual values
    actual_values = np.array([shekel_function(point) for point in test_points])
    
    # Display comparison
    logging.info(f"\n{'Point':<40} {'Actual':<12} {'Predicted':<12} {'Error':<10}")
    logging.info("-" * 80)
    
    for i, point in enumerate(test_points):
        point_str = str([f"{x:.2f}" for x in point])
        actual = actual_values[i]
        pred = predictions[i]
        error = abs(actual - pred)
        logging.info(f"{point_str:<40} {actual:<12.6f} {pred:<12.6f} {error:<10.6f}")
    
    # Calculate statistics
    mae = np.mean(np.abs(actual_values - predictions))
    mse = np.mean((actual_values - predictions)**2)
    rmse = np.sqrt(mse)
    
    logging.info("\n" + "-" * 80)
    logging.info(f"Mean Absolute Error (MAE): {mae:.6f}")
    logging.info(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    logging.info(f"Max Error: {np.max(np.abs(actual_values - predictions)):.6f}")
    
    return predictions, actual_values


def optimize_with_model(rootpath=".", model_path='shekel_model_expected.keras', bounds=None):
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
    model, scaler_X, scaler_y = load_model(rootpath, model_path)
    
    # Create objective function using the model
    def model_objective(x):
        """Objective function using neural network predictions."""
        x_scaled = scaler_X.transform(x.reshape(1, -1))
        y_pred_scaled = model.predict(x_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        return y_pred[0, 0]
    
    logging.info("\n" + "=" * 60)
    logging.info("Optimizing with Neural Network Model using SHGO")
    logging.info("=" * 60)
    
    # Run SHGO optimization
    logging.disable(logging.CRITICAL)
    result = shgo(model_objective, bounds, n=200, iters=5, sampling_method='simplicial', options={ 'ftol': 1e-3, 'minimize_every_iter': False} )
    logging.disable(logging.DEBUG)
    
    logging.info(f"\nOptimization Results (Model):")
    logging.info(f"Success: {result.success}")
    logging.info(f"Message: {result.message}")
    logging.info(f"\nOptimal solution (x*):")
    logging.info(f"  x = {result.x}")
    logging.info(f"\nOptimal function value (f(x*)):")
    logging.info(f"  f(x*) = {result.fun:.10f}")
    logging.info(f"\nNumber of function evaluations: {result.nfev}")
    logging.info(f"Number of iterations: {result.nit}")
    
    return result

def compare_optimization_results(rootpath=".", model_path='shekel_model_expected.keras'):
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
        logging.info("Error: Could not import from optimization_ex2")
        return
    
    from scipy.optimize import shgo
    
    bounds = [(0, 10)] * 4
    
    logging.info("\n" + "=" * 80)
    logging.info("COMPARING SHGO OPTIMIZATION: ACTUAL FUNCTION vs NEURAL NETWORK MODEL")
    logging.info("=" * 80)
    
    # 1. Optimize using actual Shekel function
    logging.info("\n[1/2] Running SHGO with ACTUAL Shekel function...")
    logging.info("-" * 80)
    logging.disable(logging.CRITICAL)
    result_actual = shgo(shekel_function, bounds, n=200, iters=5)
    logging.disable(logging.DEBUG)
    
    # 2. Optimize using neural network model
    logging.info("\n[2/2] Running SHGO with NEURAL NETWORK model...")
    logging.info("-" * 80)
    result_model = optimize_with_model(rootpath, model_path, bounds)
    
    # 3. Compare results
    logging.info("\n" + "=" * 80)
    logging.info("COMPARISON RESULTS")
    logging.info("=" * 80)
    
    logging.info(f"\n{'Metric':<30} {'Actual Function':<20} {'NN Model':<20} {'Difference':<15}")
    logging.info("-" * 85)
    
    # Compare optimal x values
    logging.info(f"{'Optimal X:':<30}")
    for i in range(4):
        logging.info(f"  x[{i}]{'':<24} {result_actual.x[i]:<20.6f} {result_model.x[i]:<20.6f} {abs(result_actual.x[i] - result_model.x[i]):<15.6f}")
    
    # Evaluate both solutions with actual function
    actual_at_actual = result_actual.fun
    actual_at_model = shekel_function(result_model.x)
    
    logging.info(f"\n{'Optimal f(x):':<30}")
    logging.info(f"  {'Function result':<27} {actual_at_actual:<20.10f} {result_model.fun:<20.10f} {abs(actual_at_actual - result_model.fun):<15.10f}")
    logging.info(f"  {'Actual Shekel at x*':<27} {actual_at_actual:<20.10f} {actual_at_model:<20.10f} {abs(actual_at_actual - actual_at_model):<15.10f}")
    
    logging.info(f"\n{'Performance:':<30}")
    logging.info(f"  {'Function evaluations':<27} {result_actual.nfev:<20} {result_model.nfev:<20} {result_actual.nfev - result_model.nfev:<15}")
    logging.info(f"  {'Iterations':<27} {result_actual.nit:<20} {result_model.nit:<20} {result_actual.nit - result_model.nit:<15}")
    logging.info(f"  {'Success':<27} {str(result_actual.success):<20} {str(result_model.success):<20}")
    
    # Summary
    logging.info("\n" + "=" * 80)
    logging.info("SUMMARY")
    logging.info("=" * 80)
    x_diff = np.linalg.norm(result_actual.x - result_model.x)
    f_diff = abs(actual_at_actual - actual_at_model)
    
    logging.info(f"Distance between optimal solutions (L2 norm): {x_diff:.6f}")
    logging.info(f"Difference in actual function values: {f_diff:.10f}")
    
    if f_diff < 0.01:
        logging.info("✓ Neural network model found a solution very close to the actual optimum!")
    elif f_diff < 0.1:
        logging.info("~ Neural network model found a reasonably good solution.")
    else:
        logging.info("✗ Neural network model solution differs significantly from actual optimum.")
    
    # Calculate speedup
    if result_actual.nfev > result_model.nfev:
        speedup = result_actual.nfev / result_model.nfev
        logging.info(f"⚡ Speedup: {speedup:.2f}x fewer function evaluations with NN model")
    
    return result_actual, result_model


def main(rootpath: str = ".") -> int:
    # Set random seeds for reproducibility - more comprehensive
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
     
    logging.info("=" * 60)
    
    # Demonstrate using the trained model
    logging.info("\n" + "=" * 60)
    logging.info("Testing trained model on new points:")
    logging.info("=" * 60)
    
    test_points = [
        [4.0, 4.0, 4.0, 4.0],  # Known minimum
        [1.0, 1.0, 1.0, 1.0],
        [8.0, 8.0, 8.0, 8.0],
        [3.5, 4.2, 3.8, 4.1],
        [6.0, 5.5, 6.2, 5.8]
    ]
    
    compare_model_vs_function(test_points, rootpath)
    # Compare SHGO optimization results
    result_actual, result_model = compare_optimization_results(rootpath)
    actual_results = ' '.join(str(item) for item in result_actual.x) + ' ' + str(result_actual.fun)
    model_results  = ' '.join(str(item) for item in result_model.x) + ' '  + str(result_model.fun)
    all_results = actual_results + '\n' + model_results
    return sha256(all_results.encode()).hexdigest()

if __name__ == "__main__":
    rootpath = "." if len(argv) < 2 else argv[1]
    print(main(rootpath))
