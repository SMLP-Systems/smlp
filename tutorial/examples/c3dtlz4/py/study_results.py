#!/usr/bin/python3.12
import optuna
from optuna.trial import TrialState

# Define the storage URL and study name
# Replace 'sqlite:///example.db' with your actual database path
storage_url = 'sqlite:///example.db' 
study_name = 'my_study'
from json import loads

try:
    # Load the study from the storage
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    # Get only the trials that have a 'COMPLETE' state
    # The get_trials method allows filtering by state
    complete_trials = study.get_trials(states=[TrialState.COMPLETE])

    print("N,X0,X1,X2,F1,F2,C1,C2")
    if complete_trials:
        for trial in complete_trials:
            constraints = trial.system_attrs.get("constraints", []) 
            if all(c <= 0 for c in constraints):
                print(f"{trial.number:3d},{trial.params['x0']},{trial.params['x1']},{trial.params['x2']},{trial.values[0]},{trial.values[1]},{constraints[0]},{constraints[1]}")
    else:
        print("No complete trials found for this study.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check the storage URL and study name.")


