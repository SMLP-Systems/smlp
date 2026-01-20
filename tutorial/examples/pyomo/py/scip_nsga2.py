#!/usr/bin/python3.12
"""
Multi-Objective Optimization in Python - Complete Examples
===========================================================

Since PolySCIP doesn't have a direct Python interface, this example shows
THREE better alternatives for multi-objective optimization:

1. Weighted Sum Method with SCIP/IPOPT (simple, fast)
2. Epsilon-Constraint Method with SCIP (generates Pareto front)
3. pymoo - Evolutionary Multi-Objective Optimization (NSGA-II, NSGA-III)

Prerequisites:
    pip install pyomo pymoo matplotlib numpy
    conda install -c conda-forge scip ipopt
"""

import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.environ import minimize as pyo_minimize
from pymoo.core.problem import Problem
from hashlib import sha256
from math import inf
from sys import argv

# =============================================================================
# PROBLEM DEFINITION: Bioprocess Optimization Example
# =============================================================================
# Objective 1: Maximize Yield (production output)
# Objective 2: Minimize Cost (resources consumed)
# These objectives conflict - higher yield usually requires more resources!

class BioprocessProblem(Problem):
    """
    Define multi-objective bioprocess optimization problem for pymoo
    """
    def __init__(self):
        super().__init__(
            n_var=2,  # glucose, temperature
            n_obj=2,  # yield (minimize negative), cost (minimize)
            n_constr=1,  # minimum yield constraint
            xl=np.array([0, 20]),  # lower bounds
            xu=np.array([10, 40])  # upper bounds
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate objectives and constraints
        X: array of solutions, each row is [glucose, temperature]
        """
        glucose = X[:, 0]
        temperature = X[:, 1]
        
        # Objective 1: Yield (we minimize negative yield to maximize yield)
        yield_val = glucose * (1 - ((temperature - 30)**2)/200)
        
        # Objective 2: Cost (minimize)
        cost = 2*glucose + 0.1*temperature
        
        # Output objectives (both minimized)
        out["F"] = np.column_stack([-yield_val, cost])
        
        # Constraint: minimum yield >= 3 (reformulated as g(x) <= 0)
        out["G"] = 3 - yield_val

def solve_weighted_sum(w_yield=0.5, w_cost=0.5, solver_name='ipopt'):
    """
    Combine multiple objectives into single weighted objective
    w_yield + w_cost should = 1.0
    """
    model = ConcreteModel(name="Weighted_Sum")
    
    # Decision variables
    model.glucose = Var(bounds=(0, 10), initialize=5)
    model.temperature = Var(bounds=(20, 40), initialize=30)
    
    # Objective 1: Yield (to maximize)
    def yield_expr(m):
        # Simplified yield model with optimal temperature
        return m.glucose * (1 - ((m.temperature - 30)**2)/200)
    
    # Objective 2: Cost (to minimize)
    def cost_expr(m):
        # Cost increases with both glucose and temperature
        return 2*m.glucose + 0.1*m.temperature
    
    # Normalize objectives (approximate ranges)
    max_yield_approx = 10  # Maximum possible yield
    max_cost_approx = 24   # Maximum possible cost
    
    # Combined weighted objective
    def weighted_objective(m):
        normalized_yield = yield_expr(m) / max_yield_approx
        normalized_cost = cost_expr(m) / max_cost_approx
        # Negative for maximization (yield), positive for minimization (cost)
        return -w_yield * normalized_yield + w_cost * normalized_cost
    
    model.obj = Objective(rule=weighted_objective, sense=minimize)
    
    # Constraints
    model.min_yield = Constraint(expr=yield_expr(model) >= 3)  # Minimum viable yield
    
    # Solve
    solver = SolverFactory(solver_name)
    if solver_name == 'ipopt':
        solver.options['max_iter'] = 3000
    
    results = solver.solve(model, tee=False)
    
    return {
        'glucose': model.glucose.value,
        'temperature': model.temperature.value,
        'yield': yield_expr(model),
        'cost': cost_expr(model),
        'w_yield': w_yield,
        'w_cost': w_cost
    }

def solve_epsilon_constraint(epsilon_cost, solver_name='ipopt'):
    """
    Optimize yield while constraining cost <= epsilon
    """
    model = ConcreteModel(name="Epsilon_Constraint")
    
    model.glucose = Var(bounds=(0, 10), initialize=5)
    model.temperature = Var(bounds=(20, 40), initialize=30)
    
    # Primary objective: Maximize yield
    def yield_expr(m):
        return m.glucose * (1 - ((m.temperature - 30)**2)/200)
    
    model.obj = Objective(rule=yield_expr, sense=maximize)
    
    # Secondary objective as constraint: Cost <= epsilon
    def cost_expr(m):
        return 2*m.glucose + 0.1*m.temperature
    
    model.cost_constraint = Constraint(expr=cost_expr(model) <= epsilon_cost)
    model.min_yield = Constraint(expr=yield_expr(model) >= 3)
    
    # Solve
    solver = SolverFactory(solver_name)
    if solver_name == 'ipopt':
        solver.options['max_iter'] = 3000
    
    try:
        results = solver.solve(model, tee=False)
        if results.solver.termination_condition == TerminationCondition.optimal:
            return {
                'glucose': model.glucose.value,
                'temperature': model.temperature.value,
                'yield': yield_expr(model),
                'cost': cost_expr(model),
                'epsilon': epsilon_cost,
                'feasible': True
            }
    except:
        pass
    
    return {'feasible': False, 'epsilon': epsilon_cost}

def main(rootpath: str = ".", timeout: float=5000) -> int:
    # =============================================================================
    # METHOD 1: Weighted Sum Approach
    # =============================================================================
    print("\n" + "="*70)
    print("MULTI-OBJECTIVE BIOPROCESS OPTIMIZATION")
    print("="*70)
    print("METHOD 1: Weighted Sum (Single Run)")
    print("="*70)
    print("Objectives: Maximize Yield AND Minimize Cost")
    print("="*70)
    
    # Test different weight combinations
    weights = [(0.8, 0.2), (0.5, 0.5), (0.2, 0.8)]
    weighted_solutions = []
    
    print("\nTesting different weight combinations:")
    for w_y, w_c in weights:
        sol = solve_weighted_sum(w_y, w_c)
        weighted_solutions.append(sol)
        print(f"Weights (yield={w_y}, cost={w_c}):")
        print(f"  Glucose: {sol['glucose']:.2f}, Temp: {sol['temperature']:.2f}")
        print(f"  Yield: {value(sol['yield'])}, Cost: {value(sol['cost'])}")
    
    # =============================================================================
    # METHOD 2: Epsilon-Constraint Method (Generate Pareto Front)
    # =============================================================================
    print("\n" + "="*70)
    print("METHOD 2: Epsilon-Constraint (Pareto Front Generation)")
    print("="*70)
    
    # Generate Pareto front by varying epsilon (cost constraint)
    epsilon_values = np.linspace(5, 22, 20)
    pareto_solutions = []
    
    print("\nGenerating Pareto front (varying cost constraint):")
    for eps in epsilon_values:
        sol = solve_epsilon_constraint(eps)
        if sol['feasible']:
            pareto_solutions.append(sol)
            print(f"ε={eps:.1f}: Yield={value(sol['yield'])}, Cost={value(sol['cost'])}")
    
    print(f"\nFound {len(pareto_solutions)} Pareto-optimal solutions")
    
    # =============================================================================
    # METHOD 3: pymoo - Evolutionary Multi-Objective Optimization (NSGA-II)
    # =============================================================================
    print("\n" + "="*70)
    print("METHOD 3: pymoo with NSGA-II (Evolutionary Algorithm)")
    print("="*70)
    
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.termination import get_termination
        
        
        # Create problem
        problem = BioprocessProblem()
        
        # Configure NSGA-II algorithm
        algorithm = NSGA2(
            pop_size=100,  # Population size
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        # Run optimization
        print("\nRunning NSGA-II evolutionary algorithm...")
        res = minimize(
            problem,
            algorithm,
            termination=get_termination("n_gen", 100),  # 100 generations
            seed=1,
            verbose=False
        )
        
        print(f"Found {len(res.F)} Pareto-optimal solutions")
        print(f"Best yield: {-res.F[:, 0].min():.2f}")
        print(f"Lowest cost: {res.F[:, 1].min():.2f}")
        
        # Extract solutions
        nsga2_solutions = []
        for i in range(len(res.X)):
            nsga2_solutions.append({
                'glucose': res.X[i, 0],
                'temperature': res.X[i, 1],
                'yield': -res.F[i, 0],  # Convert back from negative
                'cost': res.F[i, 1]
            })
        
        pymoo_available = True
        
    except ImportError:
        print("\npymoo not installed. Install with: pip install pymoo")
        print("Skipping NSGA-II example...")
        pymoo_available = False
        nsga2_solutions = []
    
    # =============================================================================
    # VISUALIZATION: Compare All Methods
    # =============================================================================
    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE VISUALIZATION")
    print("="*70)
    
    fig = plt.figure(figsize=(16, 10))
    
    # --- Plot 1: Pareto Front Comparison ---
    ax1 = fig.add_subplot(2, 3, 1)
    
    # Plot weighted sum solutions
    if weighted_solutions:
        ws_yields = [value(s['yield']) for s in weighted_solutions]
        ws_costs = [value(s['cost']) for s in weighted_solutions]
        ax1.plot(ws_costs, ws_yields, 'ro-', markersize=10, linewidth=2, 
                 label='Weighted Sum', zorder=5)
    
    # Plot epsilon-constraint solutions (Pareto front)
    if pareto_solutions:
        ec_yields = [value(s['yield']) for s in pareto_solutions]
        ec_costs = [value(s['cost']) for s in pareto_solutions]
        ax1.plot(ec_costs, ec_yields, 'bs-', markersize=8, linewidth=2, 
                 label='Epsilon-Constraint (Pareto)', alpha=0.7, zorder=4)
    
    # Plot NSGA-II solutions
    if pymoo_available and nsga2_solutions:
        ns_yields = [s['yield'] for s in nsga2_solutions]
        ns_costs = [s['cost'] for s in nsga2_solutions]
        ax1.scatter(ns_costs, ns_yields, c='green', s=50, alpha=0.6,
                    label='NSGA-II (pymoo)', zorder=3)
    
    ax1.set_xlabel('Cost (minimize) →', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Yield (maximize) →', fontsize=11, fontweight='bold')
    ax1.set_title('Pareto Front Comparison\n(All Methods)', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Decision Space (Glucose vs Temperature) ---
    ax2 = fig.add_subplot(2, 3, 2)
    
    if weighted_solutions:
        ws_glucose = [s['glucose'] for s in weighted_solutions]
        ws_temp = [s['temperature'] for s in weighted_solutions]
        ax2.plot(ws_glucose, ws_temp, 'ro-', markersize=10, linewidth=2, 
                 label='Weighted Sum')
    
    if pareto_solutions:
        ec_glucose = [s['glucose'] for s in pareto_solutions]
        ec_temp = [s['temperature'] for s in pareto_solutions]
        ax2.plot(ec_glucose, ec_temp, 'bs-', markersize=8, linewidth=2, 
                 label='Epsilon-Constraint', alpha=0.7)
    
    if pymoo_available and nsga2_solutions:
        ns_glucose = [s['glucose'] for s in nsga2_solutions]
        ns_temp = [s['temperature'] for s in nsga2_solutions]
        ax2.scatter(ns_glucose, ns_temp, c='green', s=50, alpha=0.6,
                    label='NSGA-II')
    
    ax2.set_xlabel('Glucose Feed Rate', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
    ax2.set_title('Decision Space\n(Design Variables)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Trade-off Surface (3D) ---
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    
    # Create mesh for surface
    gluc_range = np.linspace(0, 10, 50)
    temp_range = np.linspace(20, 40, 50)
    G, T = np.meshgrid(gluc_range, temp_range)
    Y = G * (1 - ((T - 30)**2)/200)  # Yield
    C = 2*G + 0.1*T  # Cost
    
    ax3.plot_surface(C, Y, G, cmap='viridis', alpha=0.3, edgecolor='none')
    
    # Plot Pareto solutions on surface
    if pareto_solutions:
        ec_g = [value(s['glucose']) for s in pareto_solutions]
        ec_y = [value(s['yield']) for s in pareto_solutions]
        ec_c = [value(s['cost']) for s in pareto_solutions]
        ax3.scatter(ec_c, ec_y, ec_g, c='red', s=100, marker='o', 
                    label='Pareto Front', zorder=10)
    
    ax3.set_xlabel('Cost', fontsize=9)
    ax3.set_ylabel('Yield', fontsize=9)
    ax3.set_zlabel('Glucose', fontsize=9)
    ax3.set_title('3D Trade-off Surface', fontsize=12, fontweight='bold')
    ax3.view_init(elev=20, azim=45)
    
    # --- Plot 4: Weight Sensitivity ---
    ax4 = fig.add_subplot(2, 3, 4)
    
    if weighted_solutions:
        weights_yield = [value(s['w_yield']) for s in weighted_solutions]
        yields = [value(s['yield']) for s in weighted_solutions]
        costs = [value(s['cost']) for s in weighted_solutions]
        
        ax4_twin = ax4.twinx()
        ax4.plot(weights_yield, yields, 'b-o', linewidth=2, markersize=10, 
                 label='Yield')
        ax4_twin.plot(weights_yield, costs, 'r-s', linewidth=2, markersize=10, 
                      label='Cost')
        
        ax4.set_xlabel('Weight on Yield', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Yield', fontsize=11, fontweight='bold', color='blue')
        ax4_twin.set_ylabel('Cost', fontsize=11, fontweight='bold', color='red')
        ax4.set_title('Weight Sensitivity Analysis', fontsize=12, fontweight='bold')
        ax4.tick_params(axis='y', labelcolor='blue')
        ax4_twin.tick_params(axis='y', labelcolor='red')
        ax4.grid(True, alpha=0.3)
    
    # --- Plot 5: Pareto Front Distribution ---
    ax5 = fig.add_subplot(2, 3, 5)
    
    if pareto_solutions:
        yields = np.array([value(s['yield']) for s in pareto_solutions])
        costs = np.array([value(s['cost']) for s in pareto_solutions])
        
        # Calculate distances between consecutive points
        distances = np.sqrt(np.diff(costs)**2 + np.diff(yields)**2)
        
        ax5.bar(range(len(distances)), distances, color='steelblue', alpha=0.7)
        ax5.axhline(y=distances.mean(), color='red', linestyle='--', 
                    label=f'Mean: {distances.mean():.3f}')
        ax5.set_xlabel('Solution Index', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Distance to Next Point', fontsize=11, fontweight='bold')
        ax5.set_title('Pareto Front Spacing\n(Uniformity)', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    
    # --- Plot 6: Summary Table ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║    MULTI-OBJECTIVE OPTIMIZATION: METHOD COMPARISON       ║
    ╚══════════════════════════════════════════════════════════╝
    
    METHOD 1: Weighted Sum
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ✓ Simple and fast
    ✓ Works with any solver (SCIP, IPOPT)
    ✓ Generates {len(weighted_solutions)} solutions tested
    ✗ Requires weight selection
    ✗ May miss non-convex Pareto regions
    
    Best solution: 
      Yield: {value(max(weighted_solutions, key=lambda x: value(x['yield']))['yield'])}
      Cost:  {value(min(weighted_solutions, key=lambda x: value(x['cost']))['cost'])}
    
    METHOD 2: Epsilon-Constraint
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ✓ Generates complete Pareto front
    ✓ No weight selection needed
    ✓ Found {len(pareto_solutions)} Pareto-optimal solutions
    ✗ Multiple optimization runs required
    ✗ Need to choose epsilon range
    
    Pareto front range:
      Yield: {min([value(s['yield']) for s in pareto_solutions]):.2f} - {max([value(s['yield']) for s in pareto_solutions]):.2f}
      Cost:  {min([value(s['cost']) for s in pareto_solutions]):.2f} - {max([value(s['cost']) for s in pareto_solutions]):.2f}
    
    METHOD 3: NSGA-II (pymoo)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    
    if pymoo_available:
        summary_text += f"""✓ Finds diverse Pareto set
    ✓ Handles non-convex regions
    ✓ Found {len(nsga2_solutions)} solutions
    ✗ Stochastic (results vary)
    ✗ No global optimality guarantee
    
    Solutions found:
      Best yield: {max([s['yield'] for s in nsga2_solutions]):.2f}
      Lowest cost: {min([s['cost'] for s in nsga2_solutions]):.2f}
    """
    else:
        summary_text += """✗ Not installed (pip install pymoo)
    """
    
    summary_text += """
    ══════════════════════════════════════════════════════════════
    
    RECOMMENDATIONS:
    ───────────────────────────────────────────────────────────────
    • Quick single solution → Weighted Sum
    • Complete Pareto front → Epsilon-Constraint
    • Complex non-convex → NSGA-II (pymoo)
    • Production use → Epsilon-Constraint + SCIP
    """
    
    ax6.text(0.05, 0.95, summary_text, fontsize=9, family='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('multi_objective_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'multi_objective_comparison.png'")
    if not inf == timeout:
        timer = fig.canvas.new_timer(interval=timeout, callbacks=[(plt.close, [], {})])
        timer.start()
    plt.show()
    print("\n" + "="*70)
    print("MULTI-OBJECTIVE OPTIMIZATION COMPLETE!")
    print("="*70)
    print("\nAll three methods successfully demonstrated:")
    print("1. Weighted Sum - Fast single solutions")
    print("2. Epsilon-Constraint - Complete Pareto front")
    print("3. NSGA-II (pymoo) - Evolutionary approach")
    print("="*70)
    return sha256(summary_text.encode()).hexdigest()
    
if __name__ == "__main__":
    rootpath = "." if len(argv) < 2 else argv[1]
    timeout = inf if len(argv) < 3 else argv[2]
    print(main(rootpath,timeout))
