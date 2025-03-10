
import subprocess
from flask import Flask, render_template, request, redirect, url_for, session
import os
import sys
import uuid
from flask_session import Session

# Flask App Setup
app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key')  
app.config['SESSION_TYPE'] = 'filesystem' 
Session(app)


UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
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
            dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{data_file.filename}")
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
            arguments.append(additional_command)
        
        # Debugging
        print("DEBUG: Final SMLP Command ->", " ".join(arguments))
        
        output = call_smlp_api(arguments)
        session['output'] = output
        return redirect(url_for('results'))
    
    return render_template('train.html')



#  PREDICT


# !!! To be done !!!


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        model_file = request.files.get('model_file')
        new_data_file = request.files.get('new_data_file')
        save_model = request.form.get('save_model', 'f')
        model_name = request.form.get('model_name', 'my_model')

        model_path = None
        if model_file and model_file.filename:
            filename = f"{uuid.uuid4()}_{model_file.filename}"
            model_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            model_file.save(model_path)

        newdata_path = None
        if new_data_file and new_data_file.filename:
            filename = f"{uuid.uuid4()}_{new_data_file.filename}"
            newdata_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            new_data_file.save(newdata_path)

        arguments = []
        if model_path:
            base_name, ext = os.path.splitext(model_path)
            arguments += ["-use_model", "t", "-model_name", base_name]

        if newdata_path:
            base_new_name, _ = os.path.splitext(newdata_path)
            arguments += ["-new_data", base_new_name]

        if save_model == 't':
            arguments += ["-save_model", "t", "-model_name", model_name]

        output = call_smlp_api("predict", arguments)
        session['output'] = output
        return redirect(url_for('results'))

    return render_template('predict.html')




#  EXPLORATION 
@app.route('/explore', methods=['GET', 'POST'])
def explore():
    modes_list = ['certify', 'query', 'verify', 'synthesize', 'optimize', 'optsyn']

    if request.method == 'POST':
        chosen_mode = request.form.get('explore_mode', '')

        if chosen_mode not in modes_list:
            return "Error: Invalid mode selected", 400

        data_file = request.files.get('data_file')
        spec_file = request.files.get('spec_file')

        dataset_path = None
        spec_path = None

        if data_file and data_file.filename:
            dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{data_file.filename}")
            data_file.save(dataset_path)

        if spec_file and spec_file.filename:
            spec_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{spec_file.filename}")
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
            arguments.append(additional_command)

        # Debugging
        print("DEBUG: Final SMLP Command ->", " ".join(arguments))

        output = call_smlp_api(arguments)
        session['output'] = output
        return redirect(url_for('results'))

    return render_template('exploration.html', modes=modes_list)


#  DOE

#  !!! To be done !!!

@app.route('/doe', methods=['GET', 'POST'])
def doe():
    """
    Let the user pick DOE options and pass them to SMLP in 'doe' mode.
    """
    if request.method == 'POST':
        doe_spec_file = request.files.get('doe_spec_file')
        doe_algo = request.form.get('doe_algo')
        doe_num_samples = request.form.get('doe_num_samples', '')

        arguments = []
        if doe_spec_file and doe_spec_file.filename:
            specname = f"{uuid.uuid4()}_{doe_spec_file.filename}"
            spec_path = os.path.join(app.config['UPLOAD_FOLDER'], specname)
            doe_spec_file.save(spec_path)
            base_spec, _ = os.path.splitext(spec_path)
            arguments += ["-doe_spec", base_spec]

        if doe_algo:
            arguments += ["-doe_algo", doe_algo]
        if doe_num_samples:
            arguments += ["-doe_num_samples", doe_num_samples]

        output = call_smlp_api("doe", arguments)
        session['output'] = output
        return redirect(url_for('results'))

    return render_template('doe.html')

#  RESULTS
@app.route('/results')
def results():
    output = session.get('output', 'No output available yet.')
    print("\n--- RESULTS ---")
    print(output)
    return render_template('results.html', output=output)

# MAIN
if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
