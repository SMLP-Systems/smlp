#!/usr/bin/python3.12
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from sys import argv
from math import inf
from hashlib import sha256
from sklearn.tree import DecisionTreeRegressor

# ============================================================================
# DEFINE CONSTRAINT FUNCTIONS (Python functions)
# ============================================================================

def constraint_C1(x1, x2):
    """
    C1: (x1 - 5)^2 + x2^2 ≤ 25
    Returns: constraint value (≤ 0 means satisfied)
    """
    return (x1 - 5)**2 + x2**2 - 25

def constraint_C2(x1, x2):
    """
    C2: (x1 - 8)^2 + (x2 + 3)^2 ≥ 7.7
    Converted to: -[(x1 - 8)^2 + (x2 + 3)^2] + 7.7 ≤ 0
    Returns: constraint value (≤ 0 means satisfied)
    """
    return -((x1 - 8)**2 + (x2 + 3)**2) + 7.7

# ============================================================================
# DEFINE OPTIMIZATION PROBLEM FOR PYMOO
# ============================================================================

class ParetoFromCSVData(Problem):
    """
    Multi-objective optimization using ONLY CSV data
    
    Variables: X1, X2
    Objectives: F1(X1, X2) from CSV, F2(X1, X2) from CSV
    Constraints: C1, C2
    
    NO ANALYTICAL FUNCTIONS FOR OBJECTIVES - PURE DATA-DRIVEN
    """
    
    def __init__(self,csv: str = 'objectives_data.csv'):
        super().__init__(
            n_var=2,      # Two variables: x1, x2
            n_obj=2,      # Two objectives: F1, F2 (from CSV)
            n_constr=2,   # Two constraints: C1, C2
            xl=np.array([0, 0]),      # Lower bounds: 0 ≤ x1, x2
            xu=np.array([5, 3])       # Upper bounds: x1 ≤ 5, x2 ≤ 3
        )
        self.data = pd.read_csv(csv)

        print(f"\nLoaded {len(self.data)} data points from CSV:")
        print(self.data.head(10))

        # ============================================================================
        # CREATE INTERPOLATORS FOR OBJECTIVES FROM CSV DATA
        # ============================================================================

        print("\n" + "=" * 80)
        print("CREATING INTERPOLATORS FROM CSV DATA")
        print("=" * 80)

        # Extract points and values from CSV
        self.points = self.data[['X1', 'X2']].values
        self.f1_values = self.data['F1'].values
        self.f2_values = self.data['F2'].values

        # Extract features (X1, X2) and targets (F1, F2) from CSV
        X_train = self.data[['X1', 'X2']].values

        # Create and train Decision Tree Regressors
        # These use ONLY the data from CSV - no analytical functions
        max_depth=15
        min_samples_split=5
        min_samples_leaf=2
        self.f1_tree = DecisionTreeRegressor(
            max_depth=max_depth,                 # Control model complexity
            min_samples_split=min_samples_split,  # Minimum samples to split a node
            min_samples_leaf=min_samples_leaf,   # Minimum samples in leaf nodes
            random_state=42
        )

        self.f2_tree = DecisionTreeRegressor(
            max_depth=max_depth,                 
            min_samples_split=min_samples_split,  
            min_samples_leaf=min_samples_leaf,   
            random_state=42
        )

        # Train the models on CSV data
        self.f1_tree.fit(X_train, self.f1_values)
        self.f2_tree.fit(X_train, self.f2_values)

        # Calculate R² scores to assess model quality
        f1_score = self.f1_tree.score(X_train, self.f1_values)
        f2_score = self.f2_tree.score(X_train, self.f2_values)

        print("✓ Decision Tree Regressors trained from CSV data")
        print(f"  - Using {len(X_train)} data points for training")
        print(f"  - F1 model R² score: {f1_score:.4f}")
        print(f"  - F2 model R² score: {f2_score:.4f}")
        print(f"  - Max depth: {max_depth}, min samples split: {min_samples_split}, min samples leaf {min_samples_leaf}")
       
    def get_F1_from_data(self, x1, x2):
        """
        Get F1 value from CSV data using Decision Tree Regressor
        NO analytical function - purely data-driven
        """
        X_pred = np.column_stack([x1, x2])
        return self.f1_tree.predict(X_pred)

    def get_F2_from_data(self, x1, x2):
        """
        Get F2 value from CSV data using Decision Tree Regressor
        NO analytical function - purely data-driven
        """
        X_pred = np.column_stack([x1, x2])
        return self.f2_tree.predict(X_pred)
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate objectives and constraints
        
        X: 2D array where each row is [x1, x2]
        
        IMPORTANT: F1 and F2 are obtained from CSV data via interpolation
        """
        x1 = X[:, 0]
        x2 = X[:, 1]
        
        # Get objectives from CSV data (NO analytical functions!)
        f1 = self.get_F1_from_data(x1, x2)
        f2 = self.get_F2_from_data(x1, x2)
        
        # Calculate constraints
        g1 = constraint_C1(x1, x2)
        g2 = constraint_C2(x1, x2)
        
        # Store results
        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])
    
def main(rootpath: str = ".", timeout: float=5000) -> int:
    print("=" * 80)
    print("PARETO FRONT FROM CSV DATA (NO ANALYTICAL FUNCTIONS)")
    print("=" * 80)
    print("\nIMPORTANT: F1 and F2 are read ONLY from the DataFrame/CSV")
    print("No analytical functions are used for objectives!")
    
    # ============================================================================
    # RUN NSGA-II OPTIMIZATION
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("RUNNING NSGA-II OPTIMIZATION")
    print("=" * 80)
    
    # Create problem instance
    problem = ParetoFromCSVData(rootpath + "/objectives_data.csv")
    
    # Configure NSGA-II
    algorithm = NSGA2(
        pop_size=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    # Set termination
    termination = get_termination("n_gen", 200)
    
    print("\nOptimization Settings:")
    print(f"  Objectives: F1 and F2 from CSV data (interpolated)")
    print(f"  Variables: X1 ∈ [0, 5], X2 ∈ [0, 3]")
    print(f"  Constraints:")
    print(f"    C1: (x1-5)² + x2² ≤ 25")
    print(f"    C2: (x1-8)² + (x2+3)² ≥ 7.7")
    print(f"  Population: 100")
    print(f"  Generations: 200")
    print()
    
    # Run optimization
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=True
    )
    
    # ============================================================================
    # EXTRACT PARETO FRONT
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("PARETO FRONT RESULTS")
    print("=" * 80)
    
    # Get Pareto optimal solutions
    pareto_X = res.X  # Decision variables [x1, x2]
    pareto_F = res.F  # Objective values [F1, F2] from CSV data
    
    print(f"\nNumber of Pareto optimal solutions: {len(pareto_X)}")
    
    # Create DataFrame with Pareto solutions
    pareto_df = pd.DataFrame({
        'Solution': range(1, len(pareto_X) + 1),
        'X1': pareto_X[:, 0],
        'X2': pareto_X[:, 1],
        'F1': pareto_F[:, 0],
        'F2': pareto_F[:, 1]
    })
    
    # Verify constraints for each solution
    pareto_df['C1_value'] = pareto_df.apply(
        lambda row: constraint_C1(row['X1'], row['X2']), axis=1
    )
    pareto_df['C2_value'] = pareto_df.apply(
        lambda row: constraint_C2(row['X1'], row['X2']), axis=1
    )
    pareto_df['C1_satisfied'] = pareto_df['C1_value'] <= 1e-6
    pareto_df['C2_satisfied'] = pareto_df['C2_value'] <= 1e-6
    pareto_df['All_constraints_OK'] = pareto_df['C1_satisfied'] & pareto_df['C2_satisfied']
    
    print("\n" + "-" * 80)
    print("Pareto Optimal Solutions (First 15):")
    print("-" * 80)
    display_cols = ['Solution', 'X1', 'X2', 'F1', 'F2', 'All_constraints_OK']
    print(pareto_df[display_cols].head(15).to_string(index=False))
    
    print("\n" + "-" * 80)
    print("Summary Statistics:")
    print("-" * 80)
    print(f"X1 range: [{pareto_X[:, 0].min():.3f}, {pareto_X[:, 0].max():.3f}]")
    print(f"X2 range: [{pareto_X[:, 1].min():.3f}, {pareto_X[:, 1].max():.3f}]")
    print(f"F1 range: [{pareto_F[:, 0].min():.3f}, {pareto_F[:, 0].max():.3f}]")
    print(f"F2 range: [{pareto_F[:, 1].min():.3f}, {pareto_F[:, 1].max():.3f}]")
    print(f"Feasible solutions: {pareto_df['All_constraints_OK'].sum()} / {len(pareto_df)}")
    
    # Save Pareto front to CSV
    pareto_df.to_csv(rootpath + '/pareto_front_results_dt.csv', index=False)
    print(f"\n✓ Pareto front saved to 'pareto_front_results_dt.csv'")
    
    # ============================================================================
    # ANALYZE ORIGINAL CSV DATA
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("ANALYZING ORIGINAL CSV DATA")
    print("=" * 80)
    
    data = problem.data
    # Check which original data points satisfy constraints
    data['C1_value'] = data.apply(lambda row: constraint_C1(row['X1'], row['X2']), axis=1)
    data['C2_value'] = data.apply(lambda row: constraint_C2(row['X1'], row['X2']), axis=1)
    data['C1_satisfied'] = data['C1_value'] <= 1e-6
    data['C2_satisfied'] = data['C2_value'] <= 1e-6
    data['Feasible'] = data['C1_satisfied'] & data['C2_satisfied']
    
    print(f"\nOriginal CSV data points:")
    print(f"  Total points: {len(data)}")
    print(f"  Feasible points: {data['Feasible'].sum()}")
    print(f"  Infeasible points: {(~data['Feasible']).sum()}")
    
    # Skip dominated/non-dominated calculation (too slow for large datasets)
    
    # ============================================================================
    # VISUALIZE PARETO FRONT (Only 2 plots as requested)
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Pareto Front in Objective Space
    feasible_csv = data[data['Feasible']]
    infeasible_csv = data[~data['Feasible']]
    
    ax1.scatter(infeasible_csv['F1'], infeasible_csv['F2'], c='lightcoral', s=40, alpha=0.4, marker='x', label='CSV Infeasible')
    ax1.scatter(feasible_csv['F1'], feasible_csv['F2'], c='lightblue', s=40, alpha=0.5, marker='o', label='CSV Feasible')
    ax1.scatter(pareto_F[:, 0], pareto_F[:, 1], c='darkgreen', s=100, alpha=0.9, edgecolors='black', linewidths=2, marker='D', label='Optimized Pareto Front', zorder=10)
    
    # Draw Pareto curve
    sorted_indices = np.argsort(pareto_F[:, 0])
    ax1.plot(pareto_F[sorted_indices, 0], pareto_F[sorted_indices, 1], 'g--', linewidth=2, alpha=0.5, zorder=5)
    
    ax1.set_xlabel('F1', fontsize=13, fontweight='bold')
    ax1.set_ylabel('F2', fontsize=13, fontweight='bold')
    ax1.set_title('Pareto Front in Objective Space', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Pareto Front in Decision/Variable Space
    # Draw constraint regions
    x1_grid = np.linspace(0, 5, 100)
    x2_grid = np.linspace(0, 3, 100)
    X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)
    
    # Constraint C1: (x1-5)^2 + x2^2 ≤ 25
    C1_mesh = (X1_mesh - 5)**2 + X2_mesh**2
    ax2.contour(X1_mesh, X2_mesh, C1_mesh, levels=[25], colors='red', linewidths=2.5, linestyles='--')
    ax2.contourf(X1_mesh, X2_mesh, C1_mesh, levels=[0, 25], colors=['lightgreen'], alpha=0.15)
    
    # Constraint C2: (x1-8)^2 + (x2+3)^2 ≥ 7.7
    C2_mesh = (X1_mesh - 8)**2 + (X2_mesh + 3)**2
    ax2.contour(X1_mesh, X2_mesh, C2_mesh, levels=[7.7], colors='blue', linewidths=2.5, linestyles='--')
    ax2.contourf(X1_mesh, X2_mesh, C2_mesh, levels=[7.7, 100], colors=['lightblue'], alpha=0.15)
    
    # Plot CSV data points
    ax2.scatter(infeasible_csv['X1'], infeasible_csv['X2'], c='lightcoral', s=35, marker='x', alpha=0.5, label='CSV Infeasible')
    ax2.scatter(feasible_csv['X1'], feasible_csv['X2'], c='lightblue', s=35, marker='o', alpha=0.5, label='CSV Feasible')
    
    # Plot Pareto solutions
    ax2.scatter(pareto_X[:, 0], pareto_X[:, 1], c='darkgreen', s=100, alpha=0.9, edgecolors='black', linewidths=2, marker='D', label='Pareto Solutions', zorder=10)
    
    # Draw Pareto path in variable space
    ax2.plot(pareto_X[sorted_indices, 0], pareto_X[sorted_indices, 1], 'g--', linewidth=2, alpha=0.5, zorder=5)
    
    ax2.set_xlabel('X1', fontsize=13, fontweight='bold')
    ax2.set_ylabel('X2', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 3)
    ax2.set_title('Pareto Front in Variable Space\n(Feasible Region = Green ∩ Blue)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(rootpath + '/pareto_front_two_plots.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as 'pareto_front_two_plots.png'")
    if not inf == timeout:
        timer = fig.canvas.new_timer(interval=timeout, callbacks=[(plt.close, [], {})])
        timer.start()
    plt.show()
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. pareto_front_results.csv - Pareto optimal solutions")
    print("  2. pareto_front_two_plots.png - Objective and variable space plots")
    print("\n✓ F1 and F2 were obtained ONLY from CSV data (no analytical functions)")
    print("✓ Interpolation used to estimate objectives for new X1, X2 combinations")
    print("✓ The Pareto front shows optimal trade-offs based purely on your data")
    print("=" * 80)
    return sha256(pareto_df.to_string().encode()).hexdigest()
    
if __name__ == "__main__":
    rootpath = "." if len(argv) < 2 else argv[1]
    timeout = inf if len(argv) < 3 else argv[2]
    print(main(rootpath,timeout))
