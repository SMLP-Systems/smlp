#!/usr/bin/python3.12
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from sys import argv
from hashlib import sha256
from argparse import ArgumentParser

def main() -> int:
    parser = ArgumentParser()
    parser.add_argument('-p', help='Input CSV file path', default = ".")
    parser.add_argument('-d', type=int, help='Max depth for Random Forest', default = 5)
    parser.add_argument('-n', type=int, help='Number of estimators', default = 10)
    parser.add_argument('-s', type=int, help='Min samples split', default = 10)
    
    args = parser.parse_args()
    
    # Call the actual processing function
    return run_optimization(args.p, args.d, args.n, args.s)

def run_optimization(rootpath, max_depth, n_estimators, min_samples_split) -> int:
    # Load and prepare data
    df = pd.read_csv(rootpath + '/s2_rx_anonym.csv.gz', sep=',')
    
    print("Data shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    
    # Separate fixed variables (for modeling) and design variables
    fixed_vars = ['CH', 'RANK', 'Byte']
    design_vars = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']
    objectives = ['o0', 'o1']
    
    # Get all unique combinations of fixed parameters
    unique_fixed_combinations = df[fixed_vars].drop_duplicates().values
    print(f"\nNumber of unique fixed parameter combinations: {len(unique_fixed_combinations)}")
    #print("Fixed parameter combinations:")
    #print(pd.DataFrame(unique_fixed_combinations, columns=fixed_vars))
    
    # Prepare features for surrogate models
    X = df[fixed_vars + design_vars].values
    y_o0 = df['o0'].values
    y_o1 = df['o1'].values
    
    # Train surrogate models (Random Forest)
    print("\nTraining surrogate models...")
    print(f"max_depth={max_depth}; n_estimators={n_estimators}; min_samples_split={min_samples_split}")
    
    model_o0 = RandomForestRegressor(
        n_estimators=n_estimators,      # Reduced from 100 (fewer trees = much faster)
        max_depth=max_depth,          # Limit tree depth (faster training)
        min_samples_split=min_samples_split, # Prevent deep trees
        random_state=42, 
        n_jobs=-1
    )
    model_o1 = RandomForestRegressor(
        n_estimators=n_estimators,      # Reduced from 100 (fewer trees = much faster)
        max_depth=max_depth,          # Limit tree depth (faster training)
        min_samples_split=min_samples_split, # Prevent deep trees
        random_state=42, 
        n_jobs=-1
    )
    
    model_o0.fit(X, y_o0)
    model_o1.fit(X, y_o1)
    
    result = f"Model o0 R² score: {model_o0.score(X, y_o0):.4f}\n" + \
             f"Model o1 R² score: {model_o1.score(X, y_o1):.4f}"
    
    print(result)
    return sha256(result.encode()).hexdigest()
    
    # Get bounds for design variables
    design_bounds = {
        'p0': (df['p0'].min(), df['p0'].max()),
        'p1': (df['p1'].min(), df['p1'].max()),
        'p2': (df['p2'].min(), df['p2'].max()),
        'p3': (df['p3'].min(), df['p3'].max()),
        'p4': (df['p4'].min(), df['p4'].max()),
        'p5': (df['p5'].min(), df['p5'].max())
    }
    
    print(f"\nDesign variable bounds:")
    for var, bounds in design_bounds.items():
        print(f"  {var}: [{bounds[0]}, {bounds[1]}]")
    
    
    # Define the optimization problem using pymoo
    class MaxOverFixedParamsProb(Problem):
        def __init__(self):
            # Extract bounds
            xl = np.array([design_bounds[var][0] for var in design_vars])
            xu = np.array([design_bounds[var][1] for var in design_vars])
            
            super().__init__(
                n_var=4,  # 4 design variables
                n_obj=2,  # 2 objectives
                n_constr=0,  # no constraints
                xl=xl,
                xu=xu
            )
        
        def _evaluate(self, X, out, *args, **kwargs):
            # X is a 2D array where each row is a solution [p0, p1, p2, p3, p4, p5]
            n_solutions = X.shape[0]
            n_fixed_combos = len(unique_fixed_combinations)
            
            # Initialize arrays to store maximum objectives
            max_o0 = np.zeros(n_solutions)
            max_o1 = np.zeros(n_solutions)
            
            # For each solution, evaluate over all fixed parameter combinations
            for i in range(n_solutions):
                # Create input matrix: all fixed combinations with current design variables
                X_full = np.zeros((n_fixed_combos, 9))
                X_full[:, 0:3] = unique_fixed_combinations  # CH, RANK, Byte
                X_full[:, 3:9] = np.tile(X[i, :], (n_fixed_combos, 1))  # p0, p1, p2, p3, p4, p5
                
                # Predict objectives for all fixed combinations
                o0_values = model_o0.predict(X_full)
                o1_values = model_o1.predict(X_full)
                
                # Take maximum across all fixed parameter combinations
                max_o0[i] = np.max(o0_values)
                max_o1[i] = np.max(o1_values)
            
            # Stack objectives (minimize both - so we're minimizing the maximum)
            out["F"] = np.column_stack([max_o0, max_o1])
    
    
    # Create the problem instance
    problem = MaxOverFixedParamsProb()
    
    # Configure NSGA-II algorithm
    algorithm = NSGA2(
        pop_size=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    # Configure termination criterion
    termination = get_termination("n_gen", 100)
    
    # Run optimization
    print("\nRunning NSGA-II optimization...")
    print("Objective: Find Pareto front where each objective is the MAXIMUM")
    print("           over all fixed parameter combinations (CH, RANK, Byte)")
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=True
    )
    
    print(f"\nOptimization completed!")
    print(f"Number of Pareto-optimal solutions: {len(res.F)}")
    
    # Extract Pareto front
    pareto_X = res.X  # Design variables
    pareto_F = res.F  # Objective values (max over fixed params)
    
    # Sort by first objective for display
    sorted_indices = np.argsort(pareto_F[:, 0])
    pareto_X = pareto_X[sorted_indices]
    pareto_F = pareto_F[sorted_indices]
    
    # Display results
    print("\n" + "="*100)
    print("PARETO OPTIMAL SOLUTIONS")
    print("(Objectives are MAX over all fixed parameter combinations)")
    print("="*100)
    print(f"{'#':<4} {'p0':<10} {'p1':<10} {'p2':<10} {'p3':<10} {'p4':<10} {'p5':<10} {'max(o0)':<12} {'max(o1)':<12}")
    print("-"*100)
    for i in range(len(pareto_F)):
        print(f"{i+1:<4} {pareto_X[i,0]:<10.4f} {pareto_X[i,1]:<10.4f} "
              f"{pareto_X[i,2]:<10.4f} {pareto_X[i,3]:<10.4f} "
              f"{pareto_F[i,0]:<12.6f} {pareto_F[i,1]:<12.6f}")
    
    # For comparison, also compute min, mean, std over fixed params for each Pareto solution
    print("\n" + "="*100)
    print("DETAILED ANALYSIS: Statistics over all fixed parameter combinations")
    print("="*100)
    
    detailed_stats = []
    for i in range(len(pareto_X)):
        # Create input for all fixed combinations
        X_full = np.zeros((len(unique_fixed_combinations), 7))
        X_full[:, 0:3] = unique_fixed_combinations
        X_full[:, 3:7] = np.tile(pareto_X[i, :], (len(unique_fixed_combinations), 1))
        
        # Predict
        o0_vals = model_o0.predict(X_full)
        o1_vals = model_o1.predict(X_full)
        
        detailed_stats.append({
            'solution': i+1,
            'o0_min': np.min(o0_vals),
            'o0_mean': np.mean(o0_vals),
            'o0_max': np.max(o0_vals),
            'o0_std': np.std(o0_vals),
            'o1_min': np.min(o1_vals),
            'o1_mean': np.mean(o1_vals),
            'o1_max': np.max(o1_vals),
            'o1_std': np.std(o1_vals)
        })
    
    stats_df = pd.DataFrame(detailed_stats)
    print("\nObjective o0 statistics:")
    print(stats_df[['solution', 'o0_min', 'o0_mean', 'o0_max', 'o0_std']].to_string(index=False))
    print("\nObjective o1 statistics:")
    print(stats_df[['solution', 'o1_min', 'o1_mean', 'o1_max', 'o1_std']].to_string(index=False))
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Pareto front with original data
    axes[0, 0].scatter(df['o0'], df['o1'], alpha=0.3, s=30, 
                       label='Original Data', color='lightblue')
    axes[0, 0].scatter(pareto_F[:, 0], pareto_F[:, 1], c='red', s=150, 
                       marker='*', label='Pareto Front (MAX)', edgecolors='darkred', linewidths=2)
    axes[0, 0].plot(pareto_F[:, 0], pareto_F[:, 1], 'r--', alpha=0.5, linewidth=1.5)
    axes[0, 0].set_xlabel('Objective o0', fontsize=12)
    axes[0, 0].set_ylabel('Objective o1', fontsize=12)
    axes[0, 0].set_title('Pareto Front (MAX over fixed params) vs Original Data', 
                         fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Design variables heatmap
    design_var_normalized = (pareto_X - pareto_X.min(axis=0)) / (pareto_X.max(axis=0) - pareto_X.min(axis=0) + 1e-10)
    im = axes[0, 1].imshow(design_var_normalized.T, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0, 1].set_yticks(range(4))
    axes[0, 1].set_yticklabels(design_vars)
    axes[0, 1].set_xlabel('Pareto Solution Index', fontsize=12)
    axes[0, 1].set_ylabel('Design Variables', fontsize=12)
    axes[0, 1].set_title('Design Variable Values (Normalized)', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=axes[0, 1])
    cbar.set_label('Normalized Value', fontsize=10)
    
    # Plot 3: Variability in o0 across fixed params
    axes[1, 0].errorbar(range(1, len(pareto_F)+1), 
                        stats_df['o0_mean'], 
                        yerr=stats_df['o0_std'],
                        fmt='o-', capsize=5, label='Mean ± Std')
    axes[1, 0].scatter(range(1, len(pareto_F)+1), stats_df['o0_max'], 
                       marker='*', s=100, c='red', label='Max (used in Pareto)', zorder=5)
    axes[1, 0].scatter(range(1, len(pareto_F)+1), stats_df['o0_min'], 
                       marker='v', s=50, c='blue', label='Min', zorder=5)
    axes[1, 0].set_xlabel('Pareto Solution Index', fontsize=12)
    axes[1, 0].set_ylabel('Objective o0', fontsize=12)
    axes[1, 0].set_title('o0 Variability across Fixed Parameters', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Variability in o1 across fixed params
    axes[1, 1].errorbar(range(1, len(pareto_F)+1), 
                        stats_df['o1_mean'], 
                        yerr=stats_df['o1_std'],
                        fmt='o-', capsize=5, label='Mean ± Std')
    axes[1, 1].scatter(range(1, len(pareto_F)+1), stats_df['o1_max'], 
                       marker='*', s=100, c='red', label='Max (used in Pareto)', zorder=5)
    axes[1, 1].scatter(range(1, len(pareto_F)+1), stats_df['o1_min'], 
                       marker='v', s=50, c='blue', label='Min', zorder=5)
    axes[1, 1].set_xlabel('Pareto Solution Index', fontsize=12)
    axes[1, 1].set_ylabel('Objective o1', fontsize=12)
    axes[1, 1].set_title('o1 Variability across Fixed Parameters', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pareto_front_max_over_fixed.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'pareto_front_max_over_fixed.png'")
    plt.show()
    
    # Save Pareto front to CSV
    pareto_results = pd.DataFrame(
        np.hstack([pareto_X, pareto_F]),
        columns=design_vars + ['max_o0', 'max_o1']
    )
    pareto_results.to_csv('pareto_front_solutions.csv', index=False)
    print("Pareto front solutions saved to 'pareto_front_solutions.csv'")
    
    # Save detailed statistics
    stats_df.to_csv('pareto_detailed_statistics.csv', index=False)
    print("Detailed statistics saved to 'pareto_detailed_statistics.csv'")
    
    # Print summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"Pareto front based on MAXIMUM objectives over {len(unique_fixed_combinations)} fixed parameter combinations")
    print(f"Number of Pareto-optimal solutions found: {len(pareto_F)}")
    print(f"\nObjective max(o0) range: [{pareto_F[:, 0].min():.6f}, {pareto_F[:, 0].max():.6f}]")
    print(f"Objective max(o1) range: [{pareto_F[:, 1].min():.6f}, {pareto_F[:, 1].max():.6f}]")
    print("\nDesign variable ranges in Pareto set:")
    for i, var in enumerate(design_vars):
        print(f"  {var}: [{pareto_X[:, i].min():.4f}, {pareto_X[:, i].max():.4f}]")
    
    return 0
   
if __name__ == "__main__":
    print(main())
