import matplotlib.pyplot as plt
import numpy as np
import json
from typing import Union
import pandas as pd
import random
import shutil
import os
import subprocess
import plotly.graph_objects as go
from icecream import ic
from matplotlib.colors import Normalize
import torch
import sys

# Configure icecream for debugging output
ic.configureOutput(prefix=f'Debug | ', includeContext=True)

exp = str(round(random.random(), 3))  # Generate a random experiment identifier

class plot_exp:

    def __init__(self, exp=exp, setno='15'):

        self.exp = exp
        self.setno = setno
        #self.json_file = f'Set{self.setno}_{self.exp}_experiments.json'
        self.json_data_file = f'Set{self.setno}_experiments.json'
        self.Set = f'experiment_outputs/Set{self.setno}/Set_{self.setno}_'
        self.witnesses_json = f'Set{self.setno}_{self.exp}_witnesses.json'
        self.orig_csv = "smlp_toy_basic.csv"
        self.witnesses_html_path = f'Set{self.setno}_{self.exp}_witnesses.html'
        self.stable_x_original_html_path = f'Set{self.setno}_{self.exp}_stable_x_original.html'
        self.stable_x_counter_html_path = f'Set{self.setno}_{self.exp}_stable_x_counter.html'
        self.opt_out = f'Set{self.setno}_{self.exp}_optimization_output.png'
        self.source_file_1 = f'experiment_outputs/Set{self.setno}/smlp_toy_basic.csv'
        self.source_file_2 = f'experiment_outputs/Set{self.setno}/smlp_toy_basic.spec'
        self.destination_folder = '.'    
        self.spec_file = "smlp_toy_basic.spec"
        self.orig_data = pd.read_csv('smlp_toy_basic.csv')
        self.x_bounds = f"min x: {self.orig_data.iloc[:, 0].min()} max x: {self.orig_data.iloc[:, 0].max()}"
        self.y_bounds = f"min y: {self.orig_data.iloc[:, 1].min()} max y: {self.orig_data.iloc[:, 1].max()}"
        self.obj_max = self.orig_data.iloc[:,1].max()
        if len(self.orig_data.columns) > 2:
            self.z_bounds = f"min z: {self.orig_data.iloc[:, 2].min()} max z: {self.orig_data.iloc[:, 2].max()}"
            self.obj_max = self.orig_data.iloc[:,2].max()
            self.save_to_txt(self.z_bounds, key="z_bounds")
        self.orig_len = f"{len(self.orig_data)}"
        self.save_to_txt(self.orig_len, key="Length_of_dataset")
        self.save_to_txt(self.x_bounds, key="x_bounds")
        self.save_to_txt(self.y_bounds, key="y_bounds")
        self.save_to_txt(setno, key="setno")
        self.save_to_txt(exp, key="expno")
        self.save_to_txt(self.obj_max, key="Obj_max")

    def _convert_value(self, v):
        """Convert strings to proper types (int/float/bool)"""
        if isinstance(v, str):
            if v.lower() in ('t', 'true'): return True
            if v.lower() in ('f', 'false'): return False
            try: return int(v)
            except ValueError:
                try: return float(v)
                except ValueError: return v
        return v

    def save_to_txt(self, data: Union[str,int,float,dict,list], key: str):
        if isinstance(data, list) and key == 'arguments':
            arguments_dict = {}
            i = 0
            while i < len(data):
                item = data[i]
                if item.startswith('-'):
                    # Extract key (remove leading '-')
                    k = item[1:]  
                    # Get the next item as value (if exists and isn't another flag)
                    if i+1 < len(data) and not data[i+1].startswith('-'):
                        v = data[i+1]
                        i += 1  # Skip next item since we used it
                    else:
                        v = True  # Flag without value (e.g., '-verbose')
                    # Store in dictionary
                    arguments_dict[k] = self._convert_value(v)  # Helper function for type conversion
                i += 1 
        experiments = {}
        if os.path.exists(self.json_data_file):
            with open(self.json_data_file, "r") as f:
                experiments = json.load(f)
        if self.exp not in experiments:
            experiments[self.exp] = {}

        if key == 'arguments':
            experiments[self.exp][key]= arguments_dict
        else: 
            experiments[self.exp][key] = data

        with open(self.json_data_file, "w") as f:
            json.dump(experiments, f, indent=4)

    def param_changed(self, hparams_dict, algo, n):
        ic(hparams_dict)
        with open('default_params.json', 'r') as file:
            default_dict = json.load(file)
    
        default_dict = default_dict[n]
    
        # Check for changes in hyperparameters
        changed_param = {}
        if n == 2:
            for k in hparams_dict:
                if k in hparams_dict and k in default_dict:
                    if hparams_dict[k] != default_dict[k]:
                        changed_param[k] = f"{hparams_dict[k]}"
                        param = f"{hparams_dict[k]}"
                        self.save_to_txt(changed_param,key="h_params")

        else: 
            for k in hparams_dict:
                if algo in k and k in default_dict:
                    if hparams_dict[k] != default_dict[k]:
                        changed_param[k] = f"{hparams_dict[k]}"
                        param = f"{hparams_dict[k]}"
                        self.save_to_txt(changed_param,key="h_params")

    def save_time(self, t, times=[]):
        """
        Saves the time taken for training and optimization to a text file.
    
        Parameters:
        - t: Current time.
        - times: List of times, should include start and end times for training and optimization.
        """
        times.append(t)
        if len(times) == 2:
            # Calculate training time
            train_time = f"{times[1] - times[0]}"
            # Save times to text file
            self.save_to_txt(train_time,key="model_training_time")
    
        elif len(times) == 4:
            # Calculate optimization time
            syn_time = f"{times[3] - times[2]}"
            # Save times to text file
            self.save_to_txt(syn_time,key="feasibility_time")
    
        elif len(times) == 6:
            # Calculate optimization time
            opt_time = f"{times[5] - times[4]}"
            # Save times to text file
            self.save_to_txt(opt_time,key="pareto_time")

    # Plot and save prediction data
    def prediction_save(self, X_test, y_test_pred, mm_scaler_resp):
        """
        Plots test data against original data and saves the plot.
        
        Parameters:
        - X_test: Test features.
        - y_test_pred: Predicted values.
        - mm_scaler_resp: Scaler used to inverse transform predictions.
        """
        ind = X_test.index
        data_version = 'test'
        
        # Inverse transform the data
        y_test_pred = mm_scaler_resp.inverse_transform(y_test_pred)
        X_test = mm_scaler_resp.inverse_transform(X_test)
        
        # Convert to DataFrame
        y_test_pred = pd.DataFrame(y_test_pred)
        X_test = pd.DataFrame(X_test)
        
        # Load original data
        orig_file = pd.read_csv("smlp_toy_basic.csv")
        
        # Extract original and predicted data
        x = orig_file.iloc[ind, :-1]
        y = y_test_pred.iloc[:, 0]
        
        # Create prediction DataFrame
        prediction_df = pd.concat([x, y], axis=1)
        
        # Call the main plotting function
        self.plott(x, y, data_version)

    def copy_from(self):
        """
        Copies source files to the current directory.
        """
        #if not torch.cuda.is_available():
        #    ic("CUDA is not available. Exiting program.")
        #    sys.exit(1)  # Exit with a non-zero status code

        # Your CUDA-dependent code here
        ic("CUDA is available. Proceeding with the program.")
        cuda = "CUDA is available. Proceeding with the program."
        self.save_to_txt(cuda,key="cuda")
        
        #ic(torch.cuda.is_available())
        #ic(torch.version.cuda)
        #ic(torch.__version__)
        shutil.copy2(self.source_file_1, self.destination_folder)
        shutil.copy2(self.source_file_2, self.destination_folder)

    def save_to_dict(self, data, data_version):
        """
        Saves model precision data to a .json file. Appends if the file exists.
    
        Parameters:
        - data: Dictionary containing data to be saved.
        - data_version: Indicates which version of data is being saved.
        """
        if(data_version == 'test'):
            ic(data_version)

        init_data = {}
        if os.path.exists(self.witnesses_json):
            with open(self.witnesses_json, 'r') as file:
                init_data = json.load(file)
                if data_version in init_data:
                    for key in data:
                        if key not in init_data[data_version]:
                            init_data[data_version][key] = []
                        init_data[data_version][key].append(data[key])
                else:
                    init_data[data_version] = {key: [value] for key, value in data.items()}
        else:
            init_data[data_version] = {key: value for key, value in data.items()}

        with open(self.witnesses_json, 'w') as file:
            json.dump(init_data, file, indent=4)

        if data_version == 'bounds':
            ic(init_data[data_version])

    def plott(self, x, y, data_version):
        """
        Main function to create and save plots based on data version.
        
        Parameters:
        - x: X data for plotting.
        - y: Y data for plotting.
        - data_version: Version of data to determine plot type.
        """
        # Load original data
        orig_data = pd.read_csv('smlp_toy_basic.csv')
        
        if len(orig_data.columns) > 2:
            if data_version == 'optimized':
                z_orig = orig_data.iloc[:, 2]
                y_orig = orig_data.iloc[:, 1]
                x_orig = orig_data.iloc[:, 0]
                x_additional = x.iloc[:, 0]
                y_additional = x.iloc[:, 1]
                z_additional = y
                
                # Create 3D scatter plot
                fig1 = go.Figure(data=[
                    go.Scatter3d(x=x_orig, y=y_orig, z=z_orig, mode='markers', marker=dict(color=z_orig,colorscale='Viridis', name='Original data', opacity=0.25)),
                    #go.Scatter3d(x=x_orig, y=y_orig, z=z_orig, mode='markers', marker=dict(color='red', opacity=0.25), name='Original data'),
                    go.Scatter3d(x=x_additional, y=y_additional, z=z_additional, mode='markers', marker=dict(color='red'), name='Optimal value')
                ])
                
                # Update layout
                fig1.update_layout(
                    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                    title='Scatter Plot of Optimal values on Original dataset'
                )
                
                # Save figure as HTML
                fig1.write_html(f"Set{self.setno}_{self.exp}_scatter_plot_optimized.html")
            
            else:

                z = y
                y = x.iloc[:, 1]
                x = x.iloc[:, 0]

                data = {'x': x.tolist(), 'y': y.tolist(), 'z':z.tolist()}
                self.save_to_dict(data, data_version)

                # Create 3D scatter plot
                fig0 = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5,color=z,colorscale='Viridis'))])
                
                # Update layout
                fig0.update_layout(
                    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                    title='Scatter Plot of Predicted data on test set'
                )
                
                # Save figure as HTML
                fig0.write_html(f"Set{self.setno}_{self.exp}_scatter_plot_predicted.html")
        
        else:
            plt.scatter(x, y, color='#0000ff', marker='x', s=5, label='Original data')
            x = orig_data.iloc[:, 0]
            y = orig_data.iloc[:, 1]
            plt.scatter(x, y, color='#ff0000', marker='o', s=2, label='Model Reconstruction', alpha=0.9)
            plt.xlabel('X')
            plt.ylabel('Y', rotation=0)
            plt.title('Scatter Plot representing Original data and Model Reconstruction of Original Data/Graph')
            plt.grid(True)
            plt.legend()
            plt.savefig(f"{self.setno}_{self.exp}_{data_version}.png")

    def unscale(self, b):
        data_version = 'bounds'
        min_value = self.orig_data.iloc[:, 2].min()
        max_value = self.orig_data.iloc[:, 2].max()

        # Iterate over the dictionary items
        for key, value in b.items():
            # Calculate the new value
            new_value = value * (max_value - min_value) + min_value
            # Update the dictionary with the new value
            b[key] = new_value
        ic(b)
        self.save_to_dict(b, data_version)

    def witnesses(self):
        """
        Creates and saves plots based on witnesses data and original data.
        """
        orig = pd.read_csv(self.orig_csv)
    
        if os.path.exists(self.witnesses_json):
    
            with open(self.witnesses_json, 'r') as file:
                data = json.load(file)

            # Safely get 'x' data (handles missing keys, None, and invalid types)
            x_witnesses = data.get('witnesses', {}).get('x')
            y_witnesses = data.get('witnesses', {}).get('y')
            z_witnesses = data.get('witnesses', {}).get('z')

            # Count witnesses (works for lists, arrays, floats, and missing data)
            if x_witnesses is None:
                num_witnesses = 0  # No witnesses (key missing or value=None)
            elif isinstance(x_witnesses, (float, int)):
                num_witnesses = 1  # Single numeric value = 1 witness
            elif isinstance(x_witnesses, (list, np.ndarray)):
                num_witnesses = len(x_witnesses)  # List/array = count elements
            else:
                num_witnesses = 0  # Unexpected type (e.g., str, bool)
                logging.warning(f"Ignoring invalid 'witnesses/x' type: {type(x_witnesses)}")
            
            self.save_to_txt(num_witnesses,key="number_of_witnesses")

            if z_witnesses is None:
                set_dim = 2
            else:
                set_dim = 3
            
            if set_dim == 3:

                x_orig = orig['x']
                y_orig = orig['y']
                z_orig = orig['z']

                if 'counter' in data and 'stable' in data:
                    z_counter = data['counter']['z'][:] 
                    y_counter = data['counter']['y'][:] 
                    x_counter = data['counter']['x'][:] 

                    z_sat = data['stable']['z'][:] 
                    y_sat = data['stable']['y'][:] 
                    x_sat = data['stable']['x'][:] 

                    # Create 3D scatter plot
                    fig1 = go.Figure(data=[
                        go.Scatter3d(x=x_sat, y=y_sat, z=z_sat, mode='markers', marker=dict(color='red'), name='stable witness'),
                        go.Scatter3d(x=x_counter, y=y_counter, z=z_counter, mode='markers', marker=dict(color='blue'), name='counter example'),
                        #go.Scatter3d(x=x_orig, y=y_orig, z=z_orig, mode='markers', marker=dict(colorscale='Viridis',colorbar=dict(title='Original Data Z Value'), opacity=0.5, size=3), name='original data')
                        go.Scatter3d(x=x_orig, y=y_orig, z=z_orig, mode='markers', marker=dict(size=5,color=z_orig,colorscale='Viridis', opacity=0.5),name='Original data')
                    ])
                    
                    #fig0 = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5,color=z,colorscale='Viridis'))])
                    # Update layout
                    fig1.update_layout(
                        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                        title='Scatter Plot of counter examples and stable witnesses on original data'
                    )
                    
                    # Save figure and open HTML file
                    fig1.write_html(self.stable_x_counter_html_path)
                    z = data['witnesses']['z'][:]
                    y = data['witnesses']['y'][:]
                    x = data['witnesses']['x'][:]
                    
                    # Create figure
                    fig2 = go.Figure(data=[
                        go.Scatter3d(x=x_orig, y=y_orig, z=z_orig,
                                    mode='markers',
                                    marker=dict(color=z_orig, colorscale='Viridis', opacity=0.5),
                                    name='Original data'),
                        go.Scatter3d(x=x, y=y, z=z,
                                    mode='markers',
                                    marker=dict(color=z, colorscale='Hot', size=8),
                                    name='Witnesses')
                    ])
                    
                    # Customize based on number of witnesses
                    if len(z) <= 10:
                        # Create a 10-color sequential scale from dark to light
                        colorscale = [
                            [0.0, 'rgb(5,10,172)'],    # Dark blue
                            [0.1, 'rgb(40,60,190)'],
                            [0.2, 'rgb(70,100,245)'],
                            [0.3, 'rgb(90,120,245)'],
                            [0.4, 'rgb(106,137,247)'],
                            [0.5, 'rgb(140,150,250)'], # Medium blue
                            [0.6, 'rgb(160,180,255)'],
                            [0.7, 'rgb(185,200,255)'],
                            [0.8, 'rgb(200,215,255)'],
                            [0.9, 'rgb(220,230,255)'],
                            [1.0, 'rgb(240,245,255)']  # Very light blue
                        ]
                    
                        # Normalize z values to [0,1] for color mapping
                        #z_normalized = (z - min(z)) / (max(z) - min(z)) if max(z) != min(z) else [0.5]*len(z)

                        z_np = np.array(z, dtype=float) # Convert to a NumPy array (use float for division)
                        
                        min_z = np.min(z_np) # Use NumPy's min function
                        max_z = np.max(z_np) # Use NumPy's max function
                        
                        if max_z == min_z:
                            z_normalized_np = np.full(z_np.shape, 0.5) # Create an array of 0.5s
                        else:
                            # NumPy allows direct element-wise arithmetic!
                            z_normalized_np = (z_np - min_z) / (max_z - min_z)
                        
                        # If you need the result back as a standard Python list:
                        z_normalized = z_normalized_np.tolist()
                    
                        fig2.data[1].marker.update(
                            colorscale=colorscale,
                            color=z_normalized,  # Use normalized values
                            cmin=0,
                            cmax=1,
                            size=10,  # Increased size
                            symbol='diamond',  # Different shape
                            line=dict(width=1, color='black'),  # Add border
                            opacity=0.9,
                            colorbar=dict(
                                title='Normalized Z',
                                tickvals=np.linspace(0, 1, 11),  # 11 ticks for 10 intervals
                                ticktext=[f"{min(z) + i*(max(z)-min(z))/10:.2f}" for i in range(11)]
                            )
                        )
                    
                    # Update layout
                    fig2.update_layout(
                        scene=dict(
                            xaxis_title='X',
                            yaxis_title='Y',
                            zaxis_title='Z',
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))  # Better default view
                        ),
                        title='Plot of witnesses on Original dataset'
                    )
                    
                    # Save figure
                    fig2.write_html(self.witnesses_html_path)

                    #z = data['witnesses']['z'][:] 
                    #y = data['witnesses']['y'][:] 
                    #x = data['witnesses']['x'][:] 

                    #if len(z)<=10:
                    #    colorscale = [[0, 'blue'], [0.5, 'green'], [1, 'red']]  # Custom discrete scale
                    #    
                    #    fig2.data[1].marker.update(
                    #        colorscale=colorscale,
                    #        color=z,
                    #        cmin=min(z),
                    #        cmax=max(z),
                    #        size=8  # Increase marker size
                    #    )

                    #else: 
                    #    fig2 = go.Figure(data=[
                    #        #go.Scatter3d(x=x_orig, y=y_orig, z=z_orig, mode='markers', marker=dict(color='grey', opacity=0.5), name='Original data'),
                    #        go.Scatter3d(x=x_orig, y=y_orig, z=z_orig, mode='markers', marker=dict(color=z_orig,colorscale='Viridis', opacity=0.5),name='Original data'),
                    #        go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color=z, colorscale='Hot', colorbar=dict(title='title')),name='Witnesses')
                    #    ])
                    #
                    ## Update layout
                    #fig2.update_layout(
                    #    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                    #    title='Plot of witnesses on Original dataset'
                    #)
                    #
                    ## Save figure and open HTML file
                    #fig2.write_html(self.witnesses_html_path)

                elif 'stable' in data:
                    if isinstance(data.get('stable'), dict) and 'z' in data['stable']:

                        z = data['witnesses']['z'][:] 
                        y = data['witnesses']['y'][:] 
                        x = data['witnesses']['x'][:] 

                        x_sat = data.get('stable', {}).get('x')
                        y_sat = data.get('stable', {}).get('y')
                        z_sat = data.get('stable', {}).get('z')

                        fig2 = go.Figure(data=[
                            go.Scatter3d(x=x_sat, y=y_sat, z=z_sat, mode='markers', marker=dict(color='red'), name='stable witness'),
                            go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color='blue'), name='witnesses'),
                            go.Scatter3d(x=x_orig, y=y_orig, z=z_orig, mode='markers', marker=dict(color=z_orig, opacity=0.5, colorscale='Viridis'), name='original data')
                        ])

                        # Update layout
                        fig2.update_layout(
                            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                            title='Scatter Plot of stable witnesses on Original dataset'
                        )
                        
                        # Save figure and open HTML file
                        fig2.write_html(self.stable_x_original_html_path)

                    else:
                        ic(data['counter'])

                        z = data['witnesses']['z'][:] 
                        y = data['witnesses']['y'][:] 
                        x = data['witnesses']['x'][:] 

                        x_counter = data.get('counter', {}).get('x')
                        y_counter = data.get('counter', {}).get('y')
                        z_counter = data.get('counter', {}).get('z')

                        fig2 = go.Figure(data=[
                            go.Scatter3d(x=x_counter, y=y_counter, z=z_counter, mode='markers', marker=dict(color=z_counter, colorscale='Hot', colorbar=dict(title='title')), name='Counter witnesses'),
                            go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color='blue'), name='All witnesses'),
                            go.Scatter3d(x=x_orig, y=y_orig, z=z_orig, mode='markers', marker=dict(color=z_orig, opacity=0.5, colorscale='Viridis'), name='Original data')
                            #go.Scatter3d(x=x_orig, y=y_orig, z=z_orig, mode='markers', marker=dict(color='grey', opacity=0.5), name='Original data')
                        ])

                        # Update layout
                        fig2.update_layout(
                            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                            title='Scatter Plot of counter witnesses on Original dataset'
                        )
                        
                        # Save figure and open HTML file
                        fig2.write_html(self.counter_x_original_html_path)

                else:
                    ic("no witnesses")

            else: 
                ic("data is 2d")

        else:
            ic("No witnesses to plot")

def copy_data(setno):
    """
    Copies relevant data files from source to destination folder.
    
    Parameters:
    - setno: Set number for paths.
    """
    source_folder = "."
    destination_folder = f'experiment_outputs/Set{self.setno}/'
    Set = f'experiment_outputs/Set{self.setno}/'
    source_folders = 'toy_out_dir'
    extensions = ['.png', '.html', '.txt']
    
    shutil.copytree(source_folders, Set + 'toy_out_dir', dirs_exist_ok=True)
    
    try:
        # Create the destination folder if it doesn't exist
        os.makedirs(destination_folder, exist_ok=True)
        
        # Copy files with specific extensions
        for file in os.listdir(source_folder):
            if any(file.lower().endswith(ext.lower()) for ext in extensions):
                source_file_path = os.path.join(source_folder, file)
                destination_file_path = os.path.join(destination_folder, file)
                shutil.copy2(source_file_path, destination_file_path)
                os.remove(source_file_path)
                
    except FileNotFoundError:
        print("Source folder not found.")

