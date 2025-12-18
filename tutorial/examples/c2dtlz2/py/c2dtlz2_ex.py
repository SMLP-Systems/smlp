#!/usr/bin/python3.12
import optuna
import optunahub


cdtlz = optunahub.load_module("benchmarks/dtlz_constrained")
c2dtlz2 = cdtlz.Problem(function_id=2, n_objectives=2, constraint_type=2, dimension=3)

study = optuna.create_study(
    sampler=optuna.samplers.GPSampler(seed=42, constraints_func=c2dtlz2.constraints_func, deterministic_objective=True),
    directions=c2dtlz2.directions,
)
study.optimize(c2dtlz2, n_trials=300)
optuna.visualization.plot_pareto_front(study).show()
