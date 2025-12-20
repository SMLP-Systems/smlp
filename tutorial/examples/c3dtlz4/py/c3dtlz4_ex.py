#!/usr/bin/python3.12
import optuna
import optunahub
from sys import argv
import time
import plotly.graph_objects as go
import numpy as np

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
    storage="sqlite:///example.db",
    study_name="my_study"
)
n_trials = 10 if len(argv) < 2 else int(argv[1])
print(f"Number of trials = {n_trials}")
study.optimize(c3dtlz4, n_trials, callbacks=[custom_callback])

# Create figure for decision variable space
fig = go.Figure()

# Get trials data
trials = study.best_trials
var_values = np.array([[t.params[f'x{i}'] for i in range(2)] for t in trials])

# Plot decision variable space
fig.add_trace(
    go.Scatter(
        x=var_values[:, 0],
        y=var_values[:, 1],
        mode='markers',
        name='Optimization Results',
        marker=dict(size=8, color='blue')
    )
)

# Analytical Pareto front in decision variable space (unit circle)
n_points = 100
theta = np.linspace(0, np.pi/2, n_points)
analytical_var_x1 = np.cos(theta)
analytical_var_x2 = np.sin(theta)

fig.add_trace(
    go.Scatter(
        x=analytical_var_x1,
        y=analytical_var_x2,
        mode='lines',
        name='Analytical Pareto Front',
        line=dict(color='red', width=2, dash='dash')
    )
)

# Update axes labels
fig.update_xaxes(title_text="x1")
fig.update_yaxes(title_text="x2")
fig.update_layout(
    title="Decision Variable Space (x1 vs x2)",
    height=600,
    width=700,
    showlegend=True
)

fig.show()
