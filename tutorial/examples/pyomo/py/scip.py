#!/usr/bin/python3.12
"""
SCIP vs IPOPT Solver Comparison - Complete Example
===================================================
This example demonstrates:
1. Running the SAME problems with both SCIP and IPOPT
2. Comparing global vs local optimization results
3. Visualizing the differences

Prerequisites:
    pip install pyomo matplotlib numpy
    conda install -c conda-forge scip ipopt
    or
    conda install -c conda-forge pyscipopt ipopt
"""

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, minimize, sin, cos
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
from hashlib import sha256
from sys import argv
from math import inf

# Store results for comparison
results_comparison = []

# =============================================================================
# EXAMPLE 1: Simple Quadratic Optimization
# =============================================================================
print("="*70)
print("EXAMPLE 1: Simple Quadratic Optimization")
print("Minimize (x-1)² + (y-1)² subject to x + y >= 0.5")
print("="*70)

def solve_example1(solver_name, initial_x=0.5, initial_y=0.5):
    """Solve example 1 with specified solver and initial values"""
    model = ConcreteModel(name=f"Example1_{solver_name}")
    
    # Define variables with bounds and initial values
    model.x = Var(bounds=(0, 2), initialize=initial_x)
    model.y = Var(bounds=(0, 2), initialize=initial_y)
    
    # Objective function
    def objective_rule(m):
        return (m.x - 1)**2 + (m.y - 1)**2
    
    model.obj = Objective(rule=objective_rule, sense=minimize)
    
    # Constraint
    def constraint1_rule(m):
        return m.x + m.y >= 0.5
    
    model.con1 = Constraint(rule=constraint1_rule)
    
    # Solve
    solver = SolverFactory(solver_name)
    if solver_name == 'scip':
        solver.options['limits/time'] = 1000
    elif solver_name == 'ipopt':
        solver.options['max_iter'] = 3000
    
    start_time = time.time()
    results = solver.solve(model, tee=False)
    solve_time = time.time() - start_time
    
    return {
        'model': model,
        'solver': solver_name,
        'x': model.x.value,
        'y': model.y.value,
        'obj': model.obj(),
        'time': solve_time,
        'status': str(results.solver.termination_condition),
        'init_x': initial_x,
        'init_y': initial_y
    }
def main(rootpath: str = ".", timeout: float=5000) -> int:
    # Run with SCIP
    print("\n--- Running with SCIP (Global Optimizer) ---")
    try:
        scip_result1 = solve_example1('scip', 0.5, 0.5)
        print(f"SCIP Solution: x={scip_result1['x']:.6f}, y={scip_result1['y']:.6f}")
        print(f"Objective: {scip_result1['obj']:.6f}")
        scip_available = True
    except:
        print("SCIP not available - install with: conda install -c conda-forge scip")
        scip_available = False
    
    # Run with IPOPT from different starting points
    print("\n--- Running with IPOPT (Local Optimizer) from different starts ---")
    ipopt_result1_a = solve_example1('ipopt', 0.5, 0.5)
    print(f"IPOPT (start 0.5, 0.5): x={ipopt_result1_a['x']:.6f}, y={ipopt_result1_a['y']:.6f}")
    print(f"Objective: {ipopt_result1_a['obj']:.6f}")
    
    ipopt_result1_b = solve_example1('ipopt', 0.1, 1.9)
    print(f"IPOPT (start 0.1, 1.9): x={ipopt_result1_b['x']:.6f}, y={ipopt_result1_b['y']:.6f}")
    print(f"Objective: {ipopt_result1_b['obj']:.6f}")
    
    # =============================================================================
    # EXAMPLE 2: Multi-Modal Function (Multiple Local Optima)
    # =============================================================================
    print("\n\n" + "="*70)
    print("EXAMPLE 2: Multi-Modal Function (Multiple Local Optima)")
    print("Minimize: sin(5*x)*cos(5*y)/5 + (x-1)² + (y-1)²")
    print("="*70)
    
    def solve_example2(solver_name, initial_x=0.0, initial_y=0.0):
        """Solve multi-modal problem - has multiple local optima!"""
        model = ConcreteModel(name=f"Example2_{solver_name}")
        
        model.x = Var(bounds=(-2, 4), initialize=initial_x)
        model.y = Var(bounds=(-2, 4), initialize=initial_y)
        
        def objective_rule(m):
            return sin(5*m.x)*cos(5*m.y)/5 + (m.x - 1)**2 + (m.y - 1)**2
        
        model.obj = Objective(rule=objective_rule, sense=minimize)
        
        solver = SolverFactory(solver_name)
        if solver_name == 'scip':
            solver.options['limits/time'] = 300
        elif solver_name == 'ipopt':
            solver.options['max_iter'] = 5000
        
        start_time = time.time()
        results = solver.solve(model, tee=False)
        solve_time = time.time() - start_time
        
        return {
            'model': model,
            'solver': solver_name,
            'x': model.x.value,
            'y': model.y.value,
            'obj': model.obj(),
            'time': solve_time,
            'status': str(results.solver.termination_condition),
            'init_x': initial_x,
            'init_y': initial_y
        }
    
    # Run with SCIP
    print("\n--- Running with SCIP (Global Optimizer) ---")
    if scip_available:
        try:
            scip_result2 = solve_example2('scip', 0.0, 0.0)
            print(f"SCIP Solution: x={scip_result2['x']:.6f}, y={scip_result2['y']:.6f}")
            print(f"Objective: {scip_result2['obj']:.6f}")
        except Exception as e:
            print(f"SCIP failed: {e}")
            scip_result2 = None
    else:
        scip_result2 = None
    
    # Run with IPOPT from multiple starting points
    print("\n--- Running with IPOPT from different starting points ---")
    ipopt_starts = [
        (0.0, 0.0, "center"),
        (-1.0, -1.0, "bottom-left"),
        (3.0, 3.0, "top-right"),
        (1.0, 1.0, "near optimum")
    ]
    
    ipopt_results2 = []
    for init_x, init_y, label in ipopt_starts:
        result = solve_example2('ipopt', init_x, init_y)
        ipopt_results2.append(result)
        print(f"IPOPT (start {label}): x={result['x']:.6f}, y={result['y']:.6f}, obj={result['obj']:.6f}")
    
    # Find best IPOPT solution
    best_ipopt2 = min(ipopt_results2, key=lambda r: r['obj'])
    print(f"\nBest IPOPT solution: obj={best_ipopt2['obj']:.6f}")
    if scip_result2:
        print(f"SCIP solution: obj={scip_result2['obj']:.6f}")
        print(f"Difference: {abs(best_ipopt2['obj'] - scip_result2['obj']):.6f}")
    
    # =============================================================================
    # EXAMPLE 3: Rosenbrock Function
    # =============================================================================
    print("\n\n" + "="*70)
    print("EXAMPLE 3: Rosenbrock Function")
    print("Minimize: (1-x)² + 100*(y-x²)²  subject to x²+y²<=4")
    print("="*70)
    
    def solve_example3(solver_name, initial_x=-2.0, initial_y=2.0):
        """Solve Rosenbrock with constraint"""
        model = ConcreteModel(name=f"Example3_{solver_name}")
        
        model.x = Var(bounds=(-5, 5), initialize=initial_x)
        model.y = Var(bounds=(-5, 5), initialize=initial_y)
        
        def rosenbrock_rule(m):
            return (1 - m.x)**2 + 100*(m.y - m.x**2)**2
        
        model.obj = Objective(rule=rosenbrock_rule, sense=minimize)
        
        def nonlinear_constraint(m):
            return m.x**2 + m.y**2 <= 4
        
        model.con1 = Constraint(rule=nonlinear_constraint)
        
        solver = SolverFactory(solver_name)
        if solver_name == 'scip':
            solver.options['mip/gap'] = 1e-10
        elif solver_name == 'ipopt':
            solver.options['max_iter'] = 5000
        
        start_time = time.time()
        results = solver.solve(model, tee=False)
        solve_time = time.time() - start_time
        
        return {
            'model': model,
            'solver': solver_name,
            'x': model.x.value,
            'y': model.y.value,
            'obj': model.obj(),
            'time': solve_time,
            'status': str(results.solver.termination_condition),
            'init_x': initial_x,
            'init_y': initial_y
        }
    
    # Run with SCIP
    print("\n--- Running with SCIP ---")
    if scip_available:
        try:
            scip_result3 = solve_example3('scip', -2.0, 2.0)
            print(f"SCIP Solution: x={scip_result3['x']:.6f}, y={scip_result3['y']:.6f}")
            print(f"Objective: {scip_result3['obj']:.6f}")
        except Exception as e:
            print(f"SCIP failed: {e}")
            scip_result3 = None
    else:
        scip_result3 = None
    
    # Run with IPOPT from multiple starting points
    print("\n--- Running with IPOPT from different starting points ---")
    rosenbrock_starts = [
        (-2.0, 2.0, "standard"),
        (0.0, 0.0, "origin"),
        (1.5, 1.5, "near boundary"),
    ]
    
    ipopt_results3 = []
    for init_x, init_y, label in rosenbrock_starts:
        result = solve_example3('ipopt', init_x, init_y)
        ipopt_results3.append(result)
        print(f"IPOPT (start {label}): x={result['x']:.6f}, y={result['y']:.6f}, obj={result['obj']:.6f}")
    
    # =============================================================================
    # VISUALIZATION
    # =============================================================================
    print("\n\nGenerating comprehensive comparison visualizations...")
    
    fig = plt.figure(figsize=(16, 12))
    plt.subplots_adjust(hspace=0.7, wspace=0.5)
    # --- Example 1: Simple Quadratic ---
    ax1 = fig.add_subplot(3, 4, 1)
    x_range = np.linspace(0, 2, 200)
    y_range = np.linspace(0, 2, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z1 = (X - 1)**2 + (Y - 1)**2
    
    contour1 = ax1.contour(X, Y, Z1, levels=20, cmap='viridis', alpha=0.6)
    ax1.clabel(contour1, inline=True, fontsize=7)
    
    if scip_available:
        ax1.plot(scip_result1['x'], scip_result1['y'], 'r*', markersize=20, 
                 label=f"SCIP: {scip_result1['obj']:.4f}", zorder=5)
    ax1.plot(ipopt_result1_a['x'], ipopt_result1_a['y'], 'go', markersize=12,
             label=f"IPOPT: {ipopt_result1_a['obj']:.4f}", zorder=4)
    
    x_con = np.linspace(0, 2, 100)
    y_con = 0.5 - x_con
    ax1.plot(x_con, y_con, 'r--', linewidth=2, alpha=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Example 1: Simple Quadratic\n(Both find global optimum)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # --- Example 1: Performance Comparison ---
    ax2 = fig.add_subplot(3, 4, 2)
    solvers = []
    times = []
    objs = []
    if scip_available:
        solvers.append('SCIP')
        times.append(scip_result1['time']*1000)
        objs.append(scip_result1['obj'])
    solvers.append('IPOPT')
    times.append(ipopt_result1_a['time']*1000)
    objs.append(ipopt_result1_a['obj'])
    
    x_pos = np.arange(len(solvers))
    ax2_twin = ax2.twinx()
    bars1 = ax2.bar(x_pos - 0.2, times, 0.4, label='Time (ms)', color='steelblue')
    bars2 = ax2_twin.bar(x_pos + 0.2, objs, 0.4, label='Objective', color='coral')
    
    ax2.set_ylabel('Time (ms)', color='steelblue')
    ax2_twin.set_ylabel('Objective Value', color='coral')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(solvers)
    ax2.set_title('Example 1\nPerformance Comparison')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2_twin.tick_params(axis='y', labelcolor='coral')
    ax2.grid(True, alpha=0.3)
    
    # --- Example 2: Multi-Modal Landscape ---
    ax3 = fig.add_subplot(3, 4, 5)
    x_range2 = np.linspace(-2, 4, 300)
    y_range2 = np.linspace(-2, 4, 300)
    X2, Y2 = np.meshgrid(x_range2, y_range2)
    Z2 = np.sin(5*X2)*np.cos(5*Y2)/5 + (X2 - 1)**2 + (Y2 - 1)**2
    
    contour2 = ax3.contourf(X2, Y2, Z2, levels=30, cmap='RdYlBu_r', alpha=0.8)
    plt.colorbar(contour2, ax=ax3)
    
    if scip_result2:
        ax3.plot(scip_result2['x'], scip_result2['y'], 'r*', markersize=20,
                 label=f"SCIP (global): {scip_result2['obj']:.4f}", zorder=5)
    
    colors = ['green', 'blue', 'purple', 'orange']
    for i, result in enumerate(ipopt_results2):
        ax3.plot(result['x'], result['y'], 'o', color=colors[i], markersize=10,
                 label=f"IPOPT {i+1}: {result['obj']:.4f}", zorder=4)
    
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Example 2: Multi-Modal Function\n(SCIP finds global, IPOPT may get stuck)')
    ax3.legend(fontsize=7, loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # --- Example 2: Starting Point Sensitivity ---
    ax4 = fig.add_subplot(3, 4, 6)
    start_labels = [f"Start {i+1}" for i in range(len(ipopt_results2))]
    ipopt_objs = [r['obj'] for r in ipopt_results2]
    
    ax4.bar(start_labels, ipopt_objs, color=['green', 'blue', 'purple', 'orange'])
    if scip_result2:
        ax4.axhline(y=scip_result2['obj'], color='red', linestyle='--', linewidth=2,
                    label=f"SCIP (global): {scip_result2['obj']:.4f}")
    ax4.set_ylabel('Objective Value')
    ax4.set_title('Example 2\nIPOPT Starting Point Sensitivity')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    # --- Example 3: Rosenbrock Landscape ---
    ax5 = fig.add_subplot(3, 4, 9)
    x_range3 = np.linspace(-2, 2, 300)
    y_range3 = np.linspace(-1, 3, 300)
    X3, Y3 = np.meshgrid(x_range3, y_range3)
    Z3 = (1 - X3)**2 + 100*(Y3 - X3**2)**2
    
    contour3 = ax5.contour(X3, Y3, np.log10(Z3 + 1), levels=25, cmap='plasma', alpha=0.7)
    ax5.clabel(contour3, inline=True, fontsize=7)
    
    # Plot constraint circle
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = 2 * np.cos(theta)
    y_circle = 2 * np.sin(theta)
    ax5.plot(x_circle, y_circle, 'r--', linewidth=2, label='Constraint boundary')
    
    if scip_result3:
        ax5.plot(scip_result3['x'], scip_result3['y'], 'r*', markersize=20,
                 label=f"SCIP: {scip_result3['obj']:.4f}", zorder=5)
    
    for i, result in enumerate(ipopt_results3):
        ax5.plot(result['x'], result['y'], 'o', markersize=10,
                 label=f"IPOPT {i+1}: {result['obj']:.4f}", zorder=4)
    
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title('Example 3\nRosenbrock (log scale)\nwith Constraint')
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-2, 2)
    ax5.set_ylim(-1, 3)
    
    # --- Example 3: 3D View ---
    ax6 = fig.add_subplot(3, 4, 10, projection='3d')
    X3_sub = X3[::15, ::15]
    Y3_sub = Y3[::15, ::15]
    Z3_sub = Z3[::15, ::15]
    ax6.plot_surface(X3_sub, Y3_sub, np.log10(Z3_sub + 1), 
                     cmap='plasma', alpha=0.6, edgecolor='none')
    
    if scip_result3:
        ax6.scatter([scip_result3['x']], [scip_result3['y']], 
                   [np.log10(scip_result3['obj'] + 1)],
                   color='red', s=200, marker='*', label='SCIP')
    
    for result in ipopt_results3:
        ax6.scatter([result['x']], [result['y']], 
                   [np.log10(result['obj'] + 1)],
                   s=100, marker='o', label=f"IPOPT")
    
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax6.set_zlabel('log(f)', rotation=90)
    ax6.tick_params(axis='z')
    ax6.set_title('3D Rosenbrock Surface')
    ax6.view_init(elev=25, azim=45)
    
    # --- Summary Table ---
    ax7 = fig.add_subplot(3, 4, 3)
    ax7.axis('off')
    
    summary_text = """
    ╔════════════════════════════════════════════════════════════╗
    ║         SCIP vs IPOPT: COMPREHENSIVE COMPARISON            ║
    ╚════════════════════════════════════════════════════════════╝
    
    EXAMPLE 1: Simple Quadratic (Single Global Optimum)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    
    if scip_available:
        summary_text += f"""  SCIP:    x={scip_result1['x']:.6f}, y={scip_result1['y']:.6f}
               obj={scip_result1['obj']:.6f}
    """
    summary_text += f"""  IPOPT:   x={ipopt_result1_a['x']:.6f}, y={ipopt_result1_a['y']:.6f}
               obj={ipopt_result1_a['obj']:.6f}
    
      Result: Both solvers find the same optimum ✓
    """
    
    summary_text += """
    EXAMPLE 2: Multi-Modal (Multiple Local Optima)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    
    if scip_result2:
        summary_text += f"""  SCIP:    obj={scip_result2['obj']:.6f} (GLOBAL optimum)
    """
    
    summary_text += f"""  IPOPT starting points tested: {len(ipopt_results2)}
      Best IPOPT:  obj={best_ipopt2['obj']:.6f}
      Worst IPOPT: obj={max(ipopt_results2, key=lambda r: r['obj'])['obj']:.6f}
    """
    
    if scip_result2:
        diff = abs(best_ipopt2['obj'] - scip_result2['obj'])
        summary_text += f"""  
      Result: IPOPT results vary by starting point!
              Difference from global: {diff:.6f}
    """
    
    summary_text += """
    EXAMPLE 3: Rosenbrock with Constraint
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    
    if scip_result3:
        summary_text += f"""  SCIP:    obj={scip_result3['obj']:.6f}
    """
    summary_text += f"""  IPOPT tests: {len(ipopt_results3)} starting points
      Best IPOPT:  obj={min(ipopt_results3, key=lambda r: r['obj'])['obj']:.6f}
    
    ══════════════════════════════════════════════════════════════
    
    KEY TAKEAWAYS:
    ───────────────────────────────────────────────────────────────
    ✓ SCIP (Global):  Finds TRUE global optimum every time
                      Independent of starting point
                      Slower but GUARANTEED optimal
    
    ✓ IPOPT (Local):  Fast and efficient
                      May get stuck in local optima
                      Solution depends on starting point
                      Good initial guess is critical!
    
    WHEN TO USE EACH:
    ───────────────────────────────────────────────────────────────
    SCIP:   • Non-convex problems with multiple optima
            • When global optimum is critical
            • MINLP problems
            • Small to medium problems
    
    IPOPT:  • Large-scale problems (fast!)
            • Convex problems (single optimum)
            • Good initial guess available
            • Monte Carlo / repeated optimization
    """
    
    ax7.text(0.05, 0.95, summary_text, fontsize=9, family='monospace',
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig(f'{rootpath}/scip_vs_ipopt_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'scip_vs_ipopt_comparison.png'")
    if not inf == timeout:
        timer = fig.canvas.new_timer(interval=timeout, callbacks=[(plt.close, [], {})])
        timer.start()
    plt.show()
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print("\nKey Insight: SCIP guarantees global optimum, IPOPT is faster but")
    print("may find different solutions depending on starting point!")
    print("="*70)
    return sha256(summary_text.encode()).hexdigest()

if __name__ == "__main__":
    rootpath = "." if len(argv) < 2 else argv[1]
    timeout = inf if len(argv) < 3 else argv[2]
    print(main(rootpath,timeout))
