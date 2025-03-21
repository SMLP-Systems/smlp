
import subprocess
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import os
import sys
import uuid
from flask_session import Session
from visualize_h5 import h5_visualization_route
from visualize_h5 import get_h5_plot  # Import the function
import shutil
import os



# Flask App Setup
app = Flask(__name__)




SESSION_DIR = './flask_session' 
if os.path.exists(SESSION_DIR):
    shutil.rmtree(SESSION_DIR)
    os.makedirs(SESSION_DIR)

app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key')  
app.config['SESSION_TYPE'] = 'filesystem' 
Session(app)

app.register_blueprint(h5_visualization_route)

UPLOAD_FOLDER = os.path.join('../regr_smlp/code/uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def call_smlp_api( argument_list):
    """
    mode: 'train', 'predict', 'verify', 'optimize', etc.
    argument_list: e.g. ["-data", "my_dataset", "-resp", "y1,y2", ...]
    """

    cmd_string = " ".join(argument_list) 


    cwd_dir = os.path.abspath("../regr_smlp/code/")
    try:
        result = subprocess.run(
            argument_list,
            capture_output=True,
            text=True,
            cwd=cwd_dir
        )
        full_output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        print("DEBUG: Command output ->", full_output)
        return result.stdout.strip() if result.returncode == 0 else full_output

    except Exception as e:
        return f"Error calling SMLP: {str(e)}"


#  HOME
@app.route('/')
def home():
    return render_template('index.html')

# TRAIN
@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        data_file = request.files.get('data_file')
        dataset_path = None
        
        if data_file and data_file.filename:
            dataset_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{data_file.filename}")
            data_file.save(dataset_path)
        
        def add_arg(flag, value):
            if value is not None and value != "":
                arguments.extend([flag, str(value)])
        
        arguments = ['python3', os.path.abspath("../src/run_smlp.py")]
        
        if not dataset_path :
            return "Error: Missing required dataset or spec file", 400

        # Required arguments
      
        add_arg("-data", os.path.abspath(dataset_path))
        add_arg("-out_dir", request.form.get('out_dir_val', './'))
        add_arg("-pref", request.form.get('pref_val', 'TestTrain'))
        add_arg("-mode", "train")
        add_arg("-model", request.form.get('model'))
        add_arg("-dt_sklearn_max_depth", request.form.get('dt_sklearn_max_depth'))
        add_arg("-mrmr_pred", request.form.get('mrmr_pred'))
        add_arg("-resp", request.form.get('resp'))
        add_arg("-feat", request.form.get('feat'))
        add_arg("-save_model", request.form.get('save_model'))
        add_arg("-model_name", request.form.get('model_name'))
        add_arg("-scale_feat", request.form.get('scale_feat'))
        add_arg("-scale_resp", request.form.get('scale_resp'))
        add_arg("-dt_sklearn_max_depth", request.form.get('dt_sklearn_max_depth'))
        add_arg("-train_split", request.form.get('train_split'))
        add_arg("-seed", request.form.get('seed_val'))
        add_arg("-plots", request.form.get('plots'))
        
        additional_command = request.form.get('additional_command')
        if additional_command:
            arguments.extend(additional_command.split())
        
        # Debugging
        print("DEBUG: Final SMLP Command ->", " ".join(arguments))
        
        output = call_smlp_api(arguments)
        session['output'] = output
        return redirect(url_for('results'))
    
    return render_template('train.html')



#  PREDICT

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        model_file = request.files.get('model_file')
        new_data_file = request.files.get('new_data_file')

        model_path = None
        newdata_path = None

        if model_file and model_file.filename:
            model_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{model_file.filename}")
            model_file.save(model_path)

        if new_data_file and new_data_file.filename:
            newdata_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{new_data_file.filename}")
            new_data_file.save(newdata_path)

        def add_arg(flag, value):
            if value is not None and value != "":
                arguments.extend([flag, str(value)])

        arguments = ['python3', os.path.abspath("../src/run_smlp.py")]

        if not model_path or not newdata_path:
            return "Error: Both model file and new data file are required", 400

        # Required
        add_arg("-mode", "predict")

        # Process paths (remove file extension)
        add_arg("-model_name", os.path.abspath(model_path))
        add_arg("-new_data", os.path.abspath(newdata_path))

        # Optional user inputs
        add_arg("-out_dir", request.form.get('out_dir_val', './'))
        add_arg("-pref", request.form.get('pref_val', 'PredictRun'))
        add_arg("-log_time", request.form.get('log_time', 'f'))
        add_arg("-plots", request.form.get('plots'))
        add_arg("-save_model", request.form.get('save_model'))
        add_arg("-model_name", request.form.get('model_name'))
        add_arg("-seed", request.form.get('seed_val'))


        additional_command = request.form.get('additional_command')
        if additional_command:
            arguments.extend(additional_command.split())

        # Debug output
        print("DEBUG: Final Predict SMLP Command ->", " ".join(arguments))

        output = call_smlp_api(arguments)
        session['output'] = output
        return redirect(url_for('results'))

    return render_template('predict.html')



def clear_old_plots():
    """Remove all previous plots before running a new exploration."""
    if os.path.exists(PLOT_SAVE_DIR):
        for filename in os.listdir(PLOT_SAVE_DIR):
            file_path = os.path.join(PLOT_SAVE_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)


#  EXPLORATION 
@app.route('/explore', methods=['GET', 'POST'])
def explore():
    modes_list = ['certify', 'query', 'verify', 'synthesize', 'optimize', 'optsyn']

    if request.method == 'POST':
        clear_old_plots() 
        chosen_mode = request.form.get('explore_mode', '')

        if chosen_mode not in modes_list:
            return "Error: Invalid mode selected", 400

        data_file = request.files.get('data_file')
        spec_file = request.files.get('spec_file')

        dataset_path = None
        spec_path = None

        if data_file and data_file.filename:
            dataset_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{data_file.filename}")
            data_file.save(dataset_path)

        if spec_file and spec_file.filename:
            spec_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{spec_file.filename}")
            spec_file.save(spec_path)

        if not dataset_path or not spec_path:
            return "Error: Missing required dataset or spec file", 400

        arguments = ['python3', os.path.abspath("../src/run_smlp.py")]

        def add_arg(flag, value):
            """Helper function to add arguments only if they are not empty."""
            if value is not None and value != "":
                arguments.extend([flag, str(value)])

        # Required arguments
        add_arg("-data", os.path.abspath(dataset_path))
        add_arg("-out_dir", request.form.get('out_dir_val', './'))
        add_arg("-pref", request.form.get('pref_val', 'Test113'))
        add_arg("-mode", chosen_mode)
        add_arg("-spec", os.path.abspath(spec_path))
        add_arg("-pareto", request.form.get('pareto'))
        add_arg("-resp", request.form.get('resp_expr'))
        add_arg("-feat", request.form.get('feat_expr'))
        add_arg("-model", request.form.get('model_expr'))
        add_arg("-dt_sklearn_max_depth", request.form.get('dt_sklearn_max_depth'))
        add_arg("-mrmr_pred", request.form.get('mrmr_pred'))
        add_arg("-epsilon", request.form.get('epsilon'))
        add_arg("-delta_rel", request.form.get('delta_rel'))
        add_arg("-save_model_config", request.form.get('save_model_config'))
        add_arg("-plots", request.form.get('plots'))
        add_arg("-log_time", request.form.get('log_time')) 
        add_arg("-seed", request.form.get('seed_val'))
        add_arg("-objv_names", request.form.get('objv_names'))
        add_arg("-objv_exprs", request.form.get('objv_exprs'))


        additional_command = request.form.get('additional_command')
        if additional_command:
            arguments.extend(additional_command.split())
        
        # checks if there should be a 3d plot
        session['use_h5'] = any('nn_keras' in arg for arg in arguments)
        session.modified = True
        # Debugging
        print("DEBUG: Final SMLP Command ->", " ".join(arguments))

        output = call_smlp_api(arguments)
        session['output'] = output
        return redirect(url_for('results'))

    return render_template('exploration.html', modes=modes_list)


#  DOE

@app.route('/doe', methods=['GET', 'POST'])
def doe():
    if request.method == 'POST':
        doe_spec_file = request.files.get('doe_spec_file')
        spec_path = None

        if doe_spec_file and doe_spec_file.filename:
            spec_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{doe_spec_file.filename}")
            doe_spec_file.save(spec_path)

        def add_arg(flag, value):
            if value is not None and value != "":
                arguments.extend([flag, str(value)])

        arguments = ['python3', os.path.abspath("../src/run_smlp.py")]

        if not spec_path:
            return "Error: Missing DOE spec file", 400

        # Required DOE mode
        add_arg("-doe_spec", os.path.abspath(spec_path))
        add_arg("-out_dir", request.form.get('out_dir_val', './'))
        add_arg("-pref", request.form.get('pref_val', 'TestDOE'))
        add_arg("-mode", "doe")
        add_arg("-doe_algo", request.form.get('doe_algo'))
        add_arg("-log_time", request.form.get('log_time', 'f'))

        additional_command = request.form.get('additional_command')
        if additional_command:
            arguments.extend(additional_command.split())

        # Debugging
        print("DEBUG: Final DOE SMLP Command ->", " ".join(arguments))

        output = call_smlp_api(arguments)
        session['output'] = output
        return redirect(url_for('results'))

    return render_template('doe.html')



# results
PLOT_SAVE_DIR = os.path.abspath("../regr_smlp/code/images/results")
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)  # Ensure the directory is created

@app.route('/results')
def results():
    output = session.get('output', 'No output available yet.')
    print("\n--- RESULTS ---")
    print(output)

    # Get list of available plot files
    plots = []
    if os.path.exists(PLOT_SAVE_DIR):
        plots = [f for f in os.listdir(PLOT_SAVE_DIR) if f.endswith(".png")]

    use_h5 = session.get('use_h5', False)
    h5_plot_html = get_h5_plot() if use_h5 else None

    return render_template('results.html', output=output, plots=plots, h5_plot_html=h5_plot_html, use_h5=use_h5)

# Route to serve images from the custom directory
@app.route('/plots/<filename>')
def serve_plot(filename):
    return send_from_directory(PLOT_SAVE_DIR, filename)

# MAIN
if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
