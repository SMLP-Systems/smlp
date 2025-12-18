#!/usr/bin/python3.12
import optuna
import optunahub


cdtlz = optunahub.load_module("benchmarks/dtlz_constrained")
c3dtlz4 = cdtlz.Problem(function_id=4, n_objectives=2, constraint_type=3, dimension=3)

study = optuna.create_study(
    sampler=optuna.samplers.GPSampler(seed=42, constraints_func=c3dtlz4.constraints_func, deterministic_objective=True),
    directions=c3dtlz4.directions,
)
study.optimize(c3dtlz4, n_trials=300)
optuna.visualization.plot_pareto_front(study).show()
