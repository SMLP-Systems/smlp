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
    p.add_argument('--results', '-r', default="results.csv")
    p.add_argument('--dimension', '-d', type=int, default=3)
    p.add_argument('--path', '-p', default=".")
    return p

def study_results(args: Namespace) -> bool:
    args_database = args.path + "/" + args.database
    if not exists(args_database):
        rprint(f"\n[red]ERROR: {basename(realpath(argv[0]))}: can't find file {args_database}\n")
        return 1
    try:
        study = optuna.load_study(study_name=args.study, storage='sqlite:///' + args_database)
        complete_trials = study.get_trials(states=[TrialState.COMPLETE])
        args_results = args.path + "/" + args.results
        with open(args_results, "w") as r:
            r.write("N,")
            [r.write(f"X{i},") for i in range(0, args.dimension)]
            r.write("F1,F2,C1,C2\n")
            if complete_trials:
                for trial in complete_trials:
                    constraints = trial.system_attrs.get("constraints", []) 
                    if all(c <= 0 for c in constraints):
                        r.write(f"{trial.number:3d},")
                        [r.write(f"{trial.params['x' + str(i)]},") for i in range(0, args.dimension)]
                        r.write(f"{trial.values[0]},{trial.values[1]},{constraints[0]},{constraints[1]}\n")
                return True
            else:
                print("No complete trials found for this study.")
                return False
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check the storage URL and study name.")
    
if __name__ == "__main__":
    exit(study_results(add_study_arguments().parse_args()))
