#!/usr/bin/python3.12
import optuna
from optuna.trial import TrialState
from sys import argv
from argparse import ArgumentParser, Namespace
from contextlib import redirect_stdout
from os.path import exists, basename, realpath
from rich import print as rprint

def add_study_arguments() -> ArgumentParser:
    p = ArgumentParser()
    p.add_argument('--database', '-db', default="example.db")
    p.add_argument('--study', '-s', default="my_study")
    return p

def study_results(args: Namespace) -> int:
    if not exists(args.database):
        rprint(f"\n[red]ERROR: {basename(realpath(argv[0]))}: can't find file {args.database}\n")
        return 1
    try:
        study = optuna.load_study(study_name=args.study, storage='sqlite:///' + args.database)
        complete_trials = study.get_trials(states=[TrialState.COMPLETE])
        print("N,X0,X1,X2,F1,F2,C1,C2")
        if complete_trials:
            for trial in complete_trials:
                constraints = trial.system_attrs.get("constraints", []) 
                if all(c <= 0 for c in constraints):
                    print(f"{trial.number:3d},{trial.params['x0']},{trial.params['x1']},{trial.params['x2']},{trial.values[0]},{trial.values[1]},{constraints[0]},{constraints[1]}")
            return 0
        else:
            print("No complete trials found for this study.")
            return 1
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check the storage URL and study name.")
    
if __name__ == "__main__":
    exit(study_results(add_study_arguments().parse_args()))
