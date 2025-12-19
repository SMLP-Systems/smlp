#!/usr/bin/python3.12
import optuna
import optunahub
from sys import argv
import time

start_cpu_time = time.process_time()
start_time = time.time()

def custom_callback(study, trial):
    # Check if the current trial number is a multiple of 10
    if (trial.number + 1 ) % 10 == 0:
        current_cpu_time = time.process_time()
        elapsed_cpu_time = current_cpu_time - start_cpu_time
        elapsed_time = time.time() - start_time
        print(f"Trial: {trial.number+1} [Elapsed time: {elapsed_time:.1f}] Elapsed CPU time: {elapsed_cpu_time:.1f} seconds")

cdtlz = optunahub.load_module("benchmarks/dtlz_constrained")
c3dtlz4 = cdtlz.Problem(function_id=4, n_objectives=2, constraint_type=3, dimension=3)

study = optuna.create_study(
    sampler=optuna.samplers.GPSampler(seed=42, constraints_func=c3dtlz4.constraints_func, deterministic_objective=True),
    directions=c3dtlz4.directions,
)
n_trials = 10 if len(argv) < 2 else int(argv[1])
print(f"Number of trials = {n_trials}")
study.optimize(c3dtlz4, n_trials, callbacks=[custom_callback])
optuna.visualization.plot_pareto_front(study).show()
