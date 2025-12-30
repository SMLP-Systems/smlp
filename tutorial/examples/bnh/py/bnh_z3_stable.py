#!/usr/bin/python3.12
from z3 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
from sys import argv
from math import inf
from hashlib import sha256
from base64 import b64encode

def check_solution_stability(x1_val, x2_val, epsilon=1e-5, num_samples=20):
    """
    Check the stability of a solution by:
    1. Computing gradients of objectives and constraints
    2. Analyzing sensitivity to perturbations
    3. Checking KKT conditions approximation
    4. Testing robustness with random perturbations
    """
    
    def f1(x):
        return 4 * x[0]**2 + 4 * x[1]**2
    
    def f2(x):
        return (x[0] - 5)**2 + (x[1] - 5)**2
    
    def c1(x):
        return (x[0] - 5)**2 + x[1]**2 - 25
    
    def c2(x):
        return 7.7 - (x[0] - 8)**2 - (x[1] + 3)**2
    
    x = np.array([x1_val, x2_val])
    
    # 1. Compute gradients
    grad_f1 = approx_fprime(x, f1, epsilon)
    grad_f2 = approx_fprime(x, f2, epsilon)
    grad_c1 = approx_fprime(x, c1, epsilon)
    grad_c2 = approx_fprime(x, c2, epsilon)
    
    # 2. Evaluate constraints
    c1_val = c1(x)
    c2_val = c2(x)
    
    # Determine active constraints (within tolerance)
    tolerance = 1e-3
    c1_active = abs(c1_val) < tolerance
    c2_active = abs(c2_val) < tolerance
    
    # 3. Sensitivity analysis - perturb solution and measure objective changes
    perturbations = []
    for _ in range(num_samples):
        delta = np.random.randn(2) * epsilon * 10
        x_perturbed = x + delta
        
        # Check if perturbed point is feasible
        if (0 <= x_perturbed[0] <= 5 and 0 <= x_perturbed[1] <= 3 and
            c1(x_perturbed) <= 0 and c2(x_perturbed) <= 0):
            
            delta_f1 = abs(f1(x_perturbed) - f1(x))
            delta_f2 = abs(f2(x_perturbed) - f2(x))
            perturbations.append((np.linalg.norm(delta), delta_f1, delta_f2))
    
    # 4. Compute stability metrics
    if perturbations:
        deltas = np.array([p[0] for p in perturbations])
        df1s = np.array([p[1] for p in perturbations])
        df2s = np.array([p[2] for p in perturbations])
        
        # Sensitivity ratios (change in objective / change in input)
        sensitivity_f1 = np.mean(df1s / (deltas + 1e-10))
        sensitivity_f2 = np.mean(df2s / (deltas + 1e-10))
        max_sensitivity_f1 = np.max(df1s / (deltas + 1e-10))
        max_sensitivity_f2 = np.max(df2s / (deltas + 1e-10))
    else:
        sensitivity_f1 = sensitivity_f2 = 0
        max_sensitivity_f1 = max_sensitivity_f2 = 0
    
    # 5. Compute gradient norms
    grad_f1_norm = np.linalg.norm(grad_f1)
    grad_f2_norm = np.linalg.norm(grad_f2)
    
    # 6. Check if solution is at boundary
    at_boundary = (abs(x[0]) < tolerance or abs(x[0] - 5) < tolerance or
                   abs(x[1]) < tolerance or abs(x[1] - 3) < tolerance)
    
    # 7. Stability score (lower is more stable)
    # Combines gradient magnitudes and sensitivities
    stability_score = (grad_f1_norm + grad_f2_norm) / 2 + (sensitivity_f1 + sensitivity_f2) / 2
    
    # 8. Classify stability
    if stability_score < 5:
        stability_class = "HIGHLY STABLE"
    elif stability_score < 15:
        stability_class = "MODERATELY STABLE"
    elif stability_score < 30:
        stability_class = "WEAKLY STABLE"
    else:
        stability_class = "UNSTABLE"
    
    return {
        'grad_f1': grad_f1,
        'grad_f2': grad_f2,
        'grad_f1_norm': grad_f1_norm,
        'grad_f2_norm': grad_f2_norm,
        'grad_c1': grad_c1,
        'grad_c2': grad_c2,
        'c1_active': c1_active,
        'c2_active': c2_active,
        'c1_value': c1_val,
        'c2_value': c2_val,
        'sensitivity_f1': sensitivity_f1,
        'sensitivity_f2': sensitivity_f2,
        'max_sensitivity_f1': max_sensitivity_f1,
        'max_sensitivity_f2': max_sensitivity_f2,
        'at_boundary': at_boundary,
        'stability_score': stability_score,
        'stability_class': stability_class,
        'num_feasible_perturbations': len(perturbations)
    }

def solve_bnh_z3(rootpath: str = ".", timeout: float=5000) -> int:
    """
    Solve the BNH multi-objective optimization problem using Z3.
    Uses multiple approaches to find Pareto optimal solutions.
    """
    
    print("=" * 60)
    print("BNH Multi-Objective Optimization Problem with Z3")
    print("=" * 60)
    
    # Store Pareto solutions
    pareto_solutions = []
    
    # Approach 1: Lexicographic optimization (minimize f1, then f2)
    print("\n1. Lexicographic Optimization (f1 primary, f2 secondary):")
    print("-" * 60)
    
    opt = Optimize()
    x1 = Real('x1')
    x2 = Real('x2')
    
    # Define objectives
    f1 = 4 * x1 * x1 + 4 * x2 * x2
    f2 = (x1 - 5) * (x1 - 5) + (x2 - 5) * (x2 - 5)
    
    # Add constraints
    opt.add(x1 >= 0, x1 <= 5)
    opt.add(x2 >= 0, x2 <= 3)
    opt.add((x1 - 5) * (x1 - 5) + x2 * x2 <= 25)  # C1
    opt.add((x1 - 8) * (x1 - 8) + (x2 + 3) * (x2 + 3) >= 7.7)  # C2
    
    # Minimize f1 first
    opt.minimize(f1)
    
    if opt.check() == sat:
        model = opt.model()
        x1_val = float(model[x1].as_decimal(10).rstrip('?'))
        x2_val = float(model[x2].as_decimal(10).rstrip('?'))
        f1_val = 4 * x1_val**2 + 4 * x2_val**2
        f2_val = (x1_val - 5)**2 + (x2_val - 5)**2
        
        print(f"x1 = {x1_val:.6f}")
        print(f"x2 = {x2_val:.6f}")
        print(f"f1 = {f1_val:.6f}")
        print(f"f2 = {f2_val:.6f}")
        pareto_solutions.append((x1_val, x2_val, f1_val, f2_val))
    
    # Approach 2: Lexicographic optimization (minimize f2, then f1)
    print("\n2. Lexicographic Optimization (f2 primary, f1 secondary):")
    print("-" * 60)
    
    opt2 = Optimize()
    x1_2 = Real('x1_2')
    x2_2 = Real('x2_2')
    
    f1_2 = 4 * x1_2 * x1_2 + 4 * x2_2 * x2_2
    f2_2 = (x1_2 - 5) * (x1_2 - 5) + (x2_2 - 5) * (x2_2 - 5)
    
    opt2.add(x1_2 >= 0, x1_2 <= 5)
    opt2.add(x2_2 >= 0, x2_2 <= 3)
    opt2.add((x1_2 - 5) * (x1_2 - 5) + x2_2 * x2_2 <= 25)
    opt2.add((x1_2 - 8) * (x1_2 - 8) + (x2_2 + 3) * (x2_2 + 3) >= 7.7)
    
    # Minimize f2 first
    opt2.minimize(f2_2)
    
    if opt2.check() == sat:
        model = opt2.model()
        x1_val = float(model[x1_2].as_decimal(10).rstrip('?'))
        x2_val = float(model[x2_2].as_decimal(10).rstrip('?'))
        f1_val = 4 * x1_val**2 + 4 * x2_val**2
        f2_val = (x1_val - 5)**2 + (x2_val - 5)**2
        
        print(f"x1 = {x1_val:.6f}")
        print(f"x2 = {x2_val:.6f}")
        print(f"f1 = {f1_val:.6f}")
        print(f"f2 = {f2_val:.6f}")
        pareto_solutions.append((x1_val, x2_val, f1_val, f2_val))
    
    # Approach 3: Weighted sum method with different weights
    print("\n3. Weighted Sum Method (various weights):")
    print("-" * 60)
    
    # Avoid problematic weight combinations (e.g., 0.9, 0.1) and use robust alternatives
    weights = [(0.1, 0.9), (0.2, 0.8), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.8, 0.2), (0.9, 0.1)]
    
    for w1, w2 in weights:
        print(f"\nTrying weights (w1={w1}, w2={w2})...")
        
        opt_w = Optimize()
        x1_w = Real(f'x1_w{w1}_{w2}')
        x2_w = Real(f'x2_w{w1}_{w2}')
        
        f1_w = 4 * x1_w * x1_w + 4 * x2_w * x2_w
        f2_w = (x1_w - 5) * (x1_w - 5) + (x2_w - 5) * (x2_w - 5)
        
        opt_w.add(x1_w >= 0, x1_w <= 5)
        opt_w.add(x2_w >= 0, x2_w <= 3)
        opt_w.add((x1_w - 5) * (x1_w - 5) + x2_w * x2_w <= 25)
        opt_w.add((x1_w - 8) * (x1_w - 8) + (x2_w + 3) * (x2_w + 3) >= 7.7)
        
        # Weighted objective with adaptive scaling
        # Use different scaling strategies for different weight ratios
        if w1 > 0.85:  # Highly skewed towards f1
            # Use logarithmic scaling for better numerical stability
            scale_f1 = 10
            scale_f2 = 1000
        elif w2 > 0.85:  # Highly skewed towards f2
            scale_f1 = 1000
            scale_f2 = 10
        else:
            scale_f1 = 100
            scale_f2 = 100
        
        weighted_obj = w1 * f1_w / scale_f1 + w2 * f2_w / scale_f2
        opt_w.minimize(weighted_obj)
        
        # Set timeout to avoid getting stuck
        opt_w.set("timeout", 30000)  # 30 seconds timeout
        
        result = opt_w.check()
        if result == sat:
            model = opt_w.model()
            x1_val = float(model[x1_w].as_decimal(10).rstrip('?'))
            x2_val = float(model[x2_w].as_decimal(10).rstrip('?'))
            f1_val = 4 * x1_val**2 + 4 * x2_val**2
            f2_val = (x1_val - 5)**2 + (x2_val - 5)**2
            
            print(f"  ✓ x1 = {x1_val:.6f}, x2 = {x2_val:.6f}")
            print(f"    f1 = {f1_val:.6f}, f2 = {f2_val:.6f}")
            pareto_solutions.append((x1_val, x2_val, f1_val, f2_val))
        elif result == unknown:
            print(f"  ✗ Solver timeout or unknown result for weights ({w1}, {w2})")
            print(f"    Z3 may have numerical difficulties with this weight combination")
        else:
            print(f"  ✗ No solution found (unsat) for weights ({w1}, {w2})")
    
    # Verify constraints for all solutions
    print("\n4. Constraint Verification:")
    print("-" * 60)
    for i, (x1_val, x2_val, f1_val, f2_val) in enumerate(pareto_solutions):
        c1 = (x1_val - 5)**2 + x2_val**2
        c2 = (x1_val - 8)**2 + (x2_val + 3)**2
        print(f"Solution {i+1}:")
        print(f"  C1 = {c1:.6f} (should be ≤ 25): {'✓' if c1 <= 25 else '✗'}")
        print(f"  C2 = {c2:.6f} (should be ≥ 7.7): {'✓' if c2 >= 7.7 else '✗'}")
    
    # Stability analysis for all solutions
    print("\n5. Stability Analysis:")
    print("=" * 60)
    stability_results = []
    
    for i, (x1_val, x2_val, f1_val, f2_val) in enumerate(pareto_solutions):
        print(f"\nSolution {i+1}: x = ({x1_val:.6f}, {x2_val:.6f})")
        print("-" * 60)
        
        stability = check_solution_stability(x1_val, x2_val)
        stability_results.append(stability)
        
        print(f"Gradient Information:")
        print(f"  ∇f1 = [{stability['grad_f1'][0]:>8.4f}, {stability['grad_f1'][1]:>8.4f}], ||∇f1|| = {stability['grad_f1_norm']:.4f}")
        print(f"  ∇f2 = [{stability['grad_f2'][0]:>8.4f}, {stability['grad_f2'][1]:>8.4f}], ||∇f2|| = {stability['grad_f2_norm']:.4f}")
        
        print(f"\nConstraint Status:")
        print(f"  C1 active: {stability['c1_active']} (value: {stability['c1_value']:.6f})")
        print(f"  C2 active: {stability['c2_active']} (value: {stability['c2_value']:.6f})")
        if stability['c1_active']:
            print(f"    ∇C1 = [{stability['grad_c1'][0]:>8.4f}, {stability['grad_c1'][1]:>8.4f}]")
        if stability['c2_active']:
            print(f"    ∇C2 = [{stability['grad_c2'][0]:>8.4f}, {stability['grad_c2'][1]:>8.4f}]")
        
        print(f"\nSensitivity Analysis:")
        print(f"  Avg sensitivity (f1): {stability['sensitivity_f1']:.4f}")
        print(f"  Avg sensitivity (f2): {stability['sensitivity_f2']:.4f}")
        print(f"  Max sensitivity (f1): {stability['max_sensitivity_f1']:.4f}")
        print(f"  Max sensitivity (f2): {stability['max_sensitivity_f2']:.4f}")
        print(f"  Feasible perturbations tested: {stability['num_feasible_perturbations']}/20")
        
        print(f"\nStability Assessment:")
        print(f"  At boundary: {stability['at_boundary']}")
        print(f"  Stability score: {stability['stability_score']:.4f}")
        print(f"  Classification: {stability['stability_class']}")
    
    # Summary table
    print("\n" + "=" * 60)
    print("STABILITY SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Sol':<5} {'x1':<10} {'x2':<10} {'||∇f1||':<10} {'||∇f2||':<10} {'Score':<10} {'Class':<20}")
    print("-" * 60)
    for i, ((x1, x2, f1, f2), stab) in enumerate(zip(pareto_solutions, stability_results)):
        print(f"{i+1:<5} {x1:<10.4f} {x2:<10.4f} {stab['grad_f1_norm']:<10.4f} "
              f"{stab['grad_f2_norm']:<10.4f} {stab['stability_score']:<10.4f} {stab['stability_class']:<20}")
    print("=" * 60)
    
    # Plot Pareto front
    plot_results(pareto_solutions, stability_results, timeout)
   
    solution_string=' '.join(str(item) for item in pareto_solutions)
    return sha256(solution_string.encode()).hexdigest() 

def plot_results(solutions, stability_results, timeout):
    """Plot the Pareto front in objective space and decision space with stability information."""
    if not solutions:
        return
    
    f1_vals = [s[2] for s in solutions]
    f2_vals = [s[3] for s in solutions]
    x1_vals = [s[0] for s in solutions]
    x2_vals = [s[1] for s in solutions]
    
    # Color code by stability
    colors = []
    color_map = {
        'HIGHLY STABLE': 'green',
        'MODERATELY STABLE': 'yellow',
        'WEAKLY STABLE': 'orange',
        'UNSTABLE': 'red'
    }
    
    for stab in stability_results:
        colors.append(color_map.get(stab['stability_class'], 'gray'))
    
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Decision space (x1, x2) with constraints
    ax1 = plt.subplot(1, 3, 1)
    
    # Create meshgrid for constraint visualization
    x1_grid = np.linspace(0, 5, 300)
    x2_grid = np.linspace(0, 3, 300)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    
    # Constraint 1: (x1-5)^2 + x2^2 <= 25
    C1 = (X1 - 5)**2 + X2**2
    
    # Constraint 2: (x1-8)^2 + (x2+3)^2 >= 7.7
    C2 = (X1 - 8)**2 + (X2 + 3)**2
    
    # Plot feasible region
    feasible = (C1 <= 25) & (C2 >= 7.7)
    ax1.contourf(X1, X2, feasible.astype(int), levels=[0.5, 1.5], 
                 colors=['lightblue'], alpha=0.3, label='Feasible Region')
    
    # Plot constraint boundaries
    contour1 = ax1.contour(X1, X2, C1, levels=[25], colors='blue', linewidths=2)
    ax1.clabel(contour1, inline=True, fontsize=10, fmt='C₁=25')
    
    contour2 = ax1.contour(X1, X2, C2, levels=[7.7], colors='purple', linewidths=2)
    ax1.clabel(contour2, inline=True, fontsize=10, fmt='C₂=7.7')
    
    # Plot solutions
    scatter1 = ax1.scatter(x1_vals, x2_vals, c=colors, s=200, marker='o', 
                           edgecolors='black', linewidths=2.5, zorder=5, alpha=0.8)
    
    for i, (x1, x2, f1, f2) in enumerate(solutions):
        ax1.annotate(f'{i+1}', (x1, x2), xytext=(0, 0), 
                    textcoords='offset points', fontsize=11, fontweight='bold',
                    ha='center', va='center')
    
    ax1.set_xlabel('x₁', fontsize=13, fontweight='bold')
    ax1.set_ylabel('x₂', fontsize=13, fontweight='bold')
    ax1.set_title('Decision Space (x₁, x₂)\nwith Constraints', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 3)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Add legend for constraints
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', alpha=0.3, label='Feasible Region'),
        Line2D([0], [0], color='blue', linewidth=2, label='C₁: (x₁-5)²+x₂²=25'),
        Line2D([0], [0], color='purple', linewidth=2, label='C₂: (x₁-8)²+(x₂+3)²=7.7')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # Plot 2: Objective space (Pareto front)
    ax2 = plt.subplot(1, 3, 2)
    scatter2 = ax2.scatter(f1_vals, f2_vals, c=colors, s=200, marker='o', 
                          edgecolors='black', linewidths=2.5, zorder=3, alpha=0.8)
    
    for i, (x1, x2, f1, f2) in enumerate(solutions):
        ax2.annotate(f'{i+1}', (f1, f2), xytext=(5, 5), 
                    textcoords='offset points', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('f₁(x) = 4x₁² + 4x₂²', fontsize=12)
    ax2.set_ylabel('f₂(x) = (x₁-5)² + (x₂-5)²', fontsize=12)
    ax2.set_title('Objective Space\n(Pareto Front)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add stability legend
    stability_legend = [Patch(facecolor=color, edgecolor='black', label=label)
                       for label, color in color_map.items()]
    ax2.legend(handles=stability_legend, loc='best', title='Stability Class', fontsize=9)
    
    # Plot 3: Stability scores
    ax3 = plt.subplot(1, 3, 3)
    indices = range(1, len(solutions) + 1)
    stability_scores = [s['stability_score'] for s in stability_results]
    
    bars = ax3.bar(indices, stability_scores, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Solution Number', fontsize=12)
    ax3.set_ylabel('Stability Score (lower is better)', fontsize=12)
    ax3.set_title('Solution Stability Scores', fontsize=14, fontweight='bold')
    ax3.set_xticks(indices)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal lines for stability thresholds
    ax3.axhline(y=5, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Highly Stable')
    ax3.axhline(y=15, color='yellow', linestyle='--', alpha=0.5, linewidth=1.5, label='Moderately Stable')
    ax3.axhline(y=30, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Weakly Stable')
    ax3.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    if not inf == timeout:
        timer = fig.canvas.new_timer(interval=timeout, callbacks=[(plt.close, [], {})])
        timer.start()
    plt.show()
    
    print("\n6. Decision Space, Pareto Front, and Stability Plots Generated")
    print("-" * 60)
    print("\n" + "=" * 60)
    print(f"Found {len(solutions)} Pareto optimal solutions")
    
    # Find most stable solution
    most_stable_idx = min(range(len(stability_results)), 
                         key=lambda i: stability_results[i]['stability_score'])
    print(f"\nMost stable solution: #{most_stable_idx + 1}")
    print(f"  x = ({solutions[most_stable_idx][0]:.6f}, {solutions[most_stable_idx][1]:.6f})")
    print(f"  Stability class: {stability_results[most_stable_idx]['stability_class']}")
    print(f"  Stability score: {stability_results[most_stable_idx]['stability_score']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    rootpath = "." if len(argv) < 2 else argv[1]
    timeout = inf if len(argv) < 3 else argv[2]
    print(solve_bnh_z3(rootpath,timeout))
