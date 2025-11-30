import os
import glob
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import tensorflow as tf  
from flask import Blueprint

h5_visualization_route = Blueprint('h5_visualization', __name__)

H5_DIRECTORY = os.path.abspath("../regr_smlp/code/results/")  
CSV_DIRECTORY = os.path.abspath("../regr_smlp/code/results/")  

def get_latest_file(directory, file_pattern):
    """Find the most recently created file in a directory that matches a pattern."""
    files = glob.glob(os.path.join(directory, file_pattern))  

    if not files:
        print(f"⚠ No {file_pattern} files found in:", directory)
        return None  

    latest_file = max(files, key=os.path.getctime)  
    print(f"✅ Latest {file_pattern} file:", latest_file)  
    return latest_file

def get_latest_h5_file():
    """Get the latest H5 model file."""
    return get_latest_file(H5_DIRECTORY, "*e.h5")

def get_latest_training_csv():
    """Get the latest training CSV file."""
    return get_latest_file(CSV_DIRECTORY, "*_training_predictions_summary.csv")

def get_feature_estimates():
    """Compute mean values for y1 and y2 from the latest training CSV."""
    csv_file = get_latest_training_csv()
    if not csv_file:
        return 0, 0 

    try:
        df = pd.read_csv(csv_file)
        y1_mean = df['y1'].mean()
        y2_mean = df['y2'].mean()
        print(f"📌 Estimated Feature 1 (y1) = {y1_mean}")
        print(f"📌 Estimated Feature 2 (y2) = {y2_mean}")
        return y1_mean, y2_mean
    except Exception as e:
        print(f"⚠ Error reading CSV file: {str(e)}")
        return 0, 0 
    

def load_model_and_generate_predictions():
    """Load the Keras model and generate predictions using y1, y2 as inputs."""
    h5_file = get_latest_h5_file()
    
    if not h5_file:
        return None, None, None  

    try:
        model = tf.keras.models.load_model(h5_file)
        print("✅ Model loaded successfully.")
        print(model.summary()) 

        y1_range = np.linspace(0, 12, 30) 
        y2_range = np.linspace(0, 10, 30)  
        Y1, Y2 = np.meshgrid(y1_range, y2_range)  

        f3, f4 = get_feature_estimates()

        feature3 = np.full_like(Y1.ravel(), f3)
        feature4 = np.full_like(Y2.ravel(), f4)
        inputs = np.c_[Y1.ravel(), Y2.ravel(), feature3, feature4]  

        predictions = model.predict(inputs)

        print(f"📌 Model Prediction Shape: {predictions.shape}")

        if predictions.shape[1] > 1:
            predictions = predictions[:, 0]  


        Z = predictions.reshape(Y1.shape)  
        return Y1, Y2, Z
    except Exception as e:
        print(f"❌ Error loading model or generating predictions: {str(e)}")
        return None, None, None 

def get_h5_plot():
    """Generate an interactive 3D surface plot from the model predictions."""
    y1, y2, z = load_model_and_generate_predictions()

    if y1 is None or y2 is None or z is None:
        return "<p style='color:red;'>⚠ Error: No valid model output found.</p>"

    pio.renderers.default = 'browser' 

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=y1, y=y2, z=z
    ))

    fig.update_layout(
        title="Interactive 3D Model Output",
        scene=dict(xaxis_title="y1", yaxis_title="y2", zaxis_title="Model Prediction (Z)"),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return pio.to_html(fig, full_html=False)
