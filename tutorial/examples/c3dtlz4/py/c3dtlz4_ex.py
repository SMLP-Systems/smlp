#!/usr/bin/python3.12
import optuna
import optunahub
from sys import argv
import time
from os import makedirs
from os.path import exists
from shutil import move
from argparse import Namespace
from study_results import study_results, add_study_arguments
from objectives_and_constraints import objectives_and_constraints
from hashlib import sha256
from plot_results import plot_results

start_cpu_time = time.process_time()
start_time = time.time()

def custom_callback(study, trial):
    if (trial.number + 1 ) % 10 == 0:
        current_cpu_time = time.process_time()
        elapsed_cpu_time = current_cpu_time - start_cpu_time
        elapsed_time = time.time() - start_time
        print(f"Trial: {trial.number+1} [Elapsed time: {elapsed_time:.1f}] Elapsed CPU time: {elapsed_cpu_time:.1f} seconds")

def run_study(args: Namespace) -> bool:
    makedirs(args.path, exist_ok=True)
    args_database = args.path + "/" + args.database
    args_results  = args.path + "/" + args.results
    print(args.study,args_database,args.n_trials)
    cdtlz = optunahub.load_module("benchmarks/dtlz_constrained")
    c3dtlz4 = cdtlz.Problem(function_id=4, n_objectives=2, constraint_type=3, dimension=3)
    if exists(args_database):
        move(args_database, args_database + ".bak")
    if exists(args_results):
        move(args_results, args_results + ".bak")

    study = optuna.create_study(
        sampler=optuna.samplers.GPSampler(seed=42, constraints_func=c3dtlz4.constraints_func, deterministic_objective=True),
        directions=c3dtlz4.directions,
        storage='sqlite:///' + args_database,
        study_name=args.study
)
    print(f"Number of trials = {args.n_trials}")
    study.optimize(c3dtlz4, args.n_trials, callbacks=[custom_callback])
    study_results(args)
    return True

def main() -> int:
    parser = add_study_arguments()
    parser.add_argument('--n_trials', '-n', type=int, default=10)
    args = parser.parse_args()
    if run_study(args):
       if study_results(args):
           plot_results(args.path + "/" + args.results, 5000)
           comparison_results=objectives_and_constraints(args.path + "/" + args.results) 
           print(comparison_results)
           return sha256(comparison_results.encode()).hexdigest()
       else:
           print("\nERROR: results analysis failed\n")
           return -1
    else:
        print("\nERROR: study failed\n")
        return -1

if __name__ == "__main__":
  print(main())
