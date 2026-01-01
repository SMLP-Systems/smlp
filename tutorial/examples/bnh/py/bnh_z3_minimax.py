#!/usr/bin/python3.12
from z3 import *
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from math import inf
from hashlib import sha256
from gc import collect
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

def check_minimax_stability(x1_val, x2_val, epsilon=0.001, num_directions=100):
    """
    Minimax stability check: finds worst-case performance degradation.
    Philosophy: Optimize for worst-case scenario to ensure robustness.
    """
    
    def f1(x):
        return 4 * x[0]**2 + 4 * x[1]**2
    
    def f2(x):
        return (x[0] - 5)**2 + (x[1] - 5)**2
    
    def c1(x):
        return (x[0] - 5)**2 + x[1]**2 - 25
    
    def c2(x):
        return 7.7 - (x[0] - 8)**2 - (x[1] + 3)**2
    
    def is_feasible(x):
        return (0 <= x[0] <= 5 and 0 <= x[1] <= 3 and 
                c1(x) <= 1e-6 and c2(x) <= 1e-6)
    
    x = np.array([x1_val, x2_val])
    f1_nominal = f1(x)
    f2_nominal = f2(x)
    
    # Step 1: Directional search for worst-case perturbations
    logging.info(f"    Searching worst-case in {num_directions} directions...")
    
    worst_case_f1_increase = 0
    worst_case_f2_increase = 0
    worst_direction_f1 = None
    worst_direction_f2 = None
    worst_point_f1 = x.copy()
    worst_point_f2 = x.copy()
    
    angles = np.linspace(0, 2*np.pi, num_directions, endpoint=False)
    
    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        # Binary search for maximum feasible perturbation in this direction
        low, high = 0, epsilon
        best_scale = 0
        
        for _ in range(15):  # Binary search iterations
            mid = (low + high) / 2
            x_test = x + mid * direction
            
            if is_feasible(x_test):
                best_scale = mid
                low = mid
            else:
                high = mid
        
        if best_scale > 0:
            x_perturbed = x + best_scale * direction
            
            f1_increase = f1(x_perturbed) - f1_nominal
            f2_increase = f2(x_perturbed) - f2_nominal
            
            if f1_increase > worst_case_f1_increase:
                worst_case_f1_increase = f1_increase
                worst_direction_f1 = direction
                worst_point_f1 = x_perturbed.copy()
            
            if f2_increase > worst_case_f2_increase:
                worst_case_f2_increase = f2_increase
                worst_direction_f2 = direction
                worst_point_f2 = x_perturbed.copy()
    
    # Step 2: Optimization-based worst-case search
    logging.info("    Optimization-based worst-case search...")
    
    def find_worst_case(objective_func, n_starts=10):
        best_worst = 0
        best_point = x.copy()
        
        for _ in range(n_starts):
            # Random starting perturbation
            x0 = np.random.randn(2) * epsilon * 0.3
            
            # Maximize objective (worst-case) subject to epsilon-ball constraint
            def neg_objective(delta):
                x_pert = x + delta
                if not is_feasible(x_pert):
                    return 1e10  # Large penalty for infeasible
                return -objective_func(x_pert)
            
            def constraint_ball(delta):
                return epsilon - np.linalg.norm(delta)
            
            try:
                result = minimize(
                    neg_objective,
                    x0,
                    method='SLSQP',
                    constraints={'type': 'ineq', 'fun': constraint_ball},
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                
                if result.success:
                    x_worst = x + result.x
                    if is_feasible(x_worst):
                        obj_increase = objective_func(x_worst) - objective_func(x)
                        if obj_increase > best_worst:
                            best_worst = obj_increase
                            best_point = x_worst.copy()
            except:
                pass
        
        return best_worst, best_point
    
    opt_worst_f1, opt_point_f1 = find_worst_case(f1)
    opt_worst_f2, opt_point_f2 = find_worst_case(f2)
    
    # Take maximum between directional and optimization searches
    if opt_worst_f1 > worst_case_f1_increase:
        worst_case_f1_increase = opt_worst_f1
        worst_point_f1 = opt_point_f1
    
    if opt_worst_f2 > worst_case_f2_increase:
        worst_case_f2_increase = opt_worst_f2
        worst_point_f2 = opt_point_f2
    
    # Step 3: Compute robustness metrics
    # Relative degradation (percentage)
    worst_case_f1_relative = worst_case_f1_increase / (f1_nominal + 1e-6)
    worst_case_f2_relative = worst_case_f2_increase / (f2_nominal + 1e-6)
    
    # Combined worst-case score
    worst_case_score = worst_case_f1_relative + worst_case_f2_relative
    
    # Robustness classification
    if worst_case_score < 0.05:
        robustness_class = "HIGHLY ROBUST"
        color = 'green'
    elif worst_case_score < 0.15:
        robustness_class = "MODERATELY ROBUST"
        color = 'yellow'
    elif worst_case_score < 0.30:
        robustness_class = "WEAKLY ROBUST"
        color = 'orange'
    else:
        robustness_class = "NOT ROBUST"
        color = 'red'
    
    logging.info(f"    ✓ Worst-case analysis complete")
    
    return {
        'worst_case_f1_increase': worst_case_f1_increase,
        'worst_case_f2_increase': worst_case_f2_increase,
        'worst_case_f1_relative': worst_case_f1_relative,
        'worst_case_f2_relative': worst_case_f2_relative,
        'worst_case_score': worst_case_score,
        'robustness_class': robustness_class,
        'color': color,
        'worst_direction_f1': worst_direction_f1,
        'worst_direction_f2': worst_direction_f2,
        'worst_point_f1': worst_point_f1,
        'worst_point_f2': worst_point_f2,
        'epsilon': epsilon,
        'nominal_f1': f1_nominal,
        'nominal_f2': f2_nominal
    }

def solve_bnh_z3_minimax(rootpath: str = ".", timeout: float = 5000) -> str:
    collect()
    """
    Solve BNH problem using Z3 with minimax stability analysis.
    """
    
    logging.info("=" * 70)
    logging.info("BNH Multi-Objective Optimization with Minimax Stability Analysis")
    logging.info("=" * 70)
    logging.info("\nMinimax Philosophy:")
    logging.info("  - Optimizes for WORST-CASE scenario")
    logging.info("  - Ensures robustness under uncertainty")
    logging.info("  - Provides strong guarantees against adversarial perturbations")
    logging.info("=" * 70)
    
    pareto_solutions = []
    
    # Approach 1: Lexicographic (f1 primary)
    logging.info("\n1. Lexicographic Optimization (f1 primary):")
    logging.info("-" * 70)
    
    opt = Optimize()
    x1, x2 = Real('x1'), Real('x2')
    f1 = 4 * x1 * x1 + 4 * x2 * x2
    f2 = (x1 - 5) * (x1 - 5) + (x2 - 5) * (x2 - 5)
    
    opt.add(x1 >= 0, x1 <= 5, x2 >= 0, x2 <= 3)
    opt.add((x1 - 5) * (x1 - 5) + x2 * x2 <= 25)
    opt.add((x1 - 8) * (x1 - 8) + (x2 + 3) * (x2 + 3) >= 7.7)
    opt.set("timeout", 10000)
    opt.minimize(f1)
    
    if opt.check() == sat:
        model = opt.model()
        x1_val = float(model[x1].as_decimal(10).rstrip('?'))
        x2_val = float(model[x2].as_decimal(10).rstrip('?'))
        f1_val = round(4 * x1_val**2 + 4 * x2_val**2, 4)
        f2_val = round((x1_val - 5)**2 + (x2_val - 5)**2, 4)
        
        logging.info(f"  Solution: x=({x1_val:.6f}, {x2_val:.6f})")
        logging.info(f"  Objectives: f1={f1_val:.4f}, f2={f2_val:.4f}")
        pareto_solutions.append((x1_val, x2_val, f1_val, f2_val))
    
    # Approach 2: Lexicographic (f2 primary)
    logging.info("\n2. Lexicographic Optimization (f2 primary):")
    logging.info("-" * 70)
    
    opt.pop()
    opt2 = Optimize()
    x1_2, x2_2 = Real('x1_2'), Real('x2_2')
    f1_2 = 4 * x1_2 * x1_2 + 4 * x2_2 * x2_2
    f2_2 = (x1_2 - 5) * (x1_2 - 5) + (x2_2 - 5) * (x2_2 - 5)
    
    opt2.add(x1_2 >= 0, x1_2 <= 5, x2_2 >= 0, x2_2 <= 3)
    opt2.add((x1_2 - 5) * (x1_2 - 5) + x2_2 * x2_2 <= 25)
    opt2.add((x1_2 - 8) * (x1_2 - 8) + (x2_2 + 3) * (x2_2 + 3) >= 7.7)
    opt2.set("timeout", 10000)
    opt2.minimize(f2_2)
    opt2.pop()
    
    if opt2.check() == sat:
        model = opt2.model()
        x1_val = float(model[x1_2].as_decimal(10).rstrip('?'))
        x2_val = float(model[x2_2].as_decimal(10).rstrip('?'))
        f1_val = round(4 * x1_val**2 + 4 * x2_val**2, 4)
        f2_val = round((x1_val - 5)**2 + (x2_val - 5)**2, 4)
        
        logging.info(f"  Solution: x=({x1_val:.6f}, {x2_val:.6f})")
        logging.info(f"  Objectives: f1={f1_val:.4f}, f2={f2_val:.4f}")
        pareto_solutions.append((x1_val, x2_val, f1_val, f2_val))
    
    # Approach 3: Weighted sum
    logging.info("\n3. Weighted Sum Method:")
    logging.info("-" * 70)
    
    weights = [(0.1, 0.9), (0.2, 0.8), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.8, 0.2), (0.9, 0.1)]
    
    for w1, w2 in weights:
        opt_w = Optimize()
        x1_w = Real(f'x1_{w1}_{w2}')
        x2_w = Real(f'x2_{w1}_{w2}')
        
        f1_w = 4 * x1_w * x1_w + 4 * x2_w * x2_w
        f2_w = (x1_w - 5) * (x1_w - 5) + (x2_w - 5) * (x2_w - 5)
        
        opt_w.add(x1_w >= 0, x1_w <= 5, x2_w >= 0, x2_w <= 3)
        opt_w.add((x1_w - 5) * (x1_w - 5) + x2_w * x2_w <= 25)
        opt_w.add((x1_w - 8) * (x1_w - 8) + (x2_w + 3) * (x2_w + 3) >= 7.7)

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

        weighted_obj = w1 * f1_w / 100 + w2 * f2_w / 100
        opt_w.set("timeout", 30000)
        opt_w.minimize(weighted_obj)
        
        if opt_w.check() == sat:
            model = opt_w.model()
            x1_val = float(model[x1_w].as_decimal(10).rstrip('?'))
            x2_val = float(model[x2_w].as_decimal(10).rstrip('?'))
            f1_val = round(4 * x1_val**2 + 4 * x2_val**2, 4)
            f2_val = round((x1_val - 5)**2 + (x2_val - 5)**2, 4)
            
            logging.info(f"  w=({w1},{w2}): x=({x1_val:.4f},{x2_val:.4f}), "
                  f"f1={f1_val:.4f}, f2={f2_val:.4f}")
            pareto_solutions.append((x1_val, x2_val, f1_val, f2_val))
            opt_w.pop()
    
    # Minimax stability analysis
    logging.info("\n" + "=" * 70)
    logging.info("MINIMAX STABILITY ANALYSIS")
    logging.info("=" * 70)
    
    minimax_results = []
    
    for i, (x1_val, x2_val, f1_val, f2_val) in enumerate(pareto_solutions):
        logging.info(f"\nSolution {i+1}: x = ({x1_val:.6f}, {x2_val:.6f})")
        logging.info("-" * 70)
        
        mm = check_minimax_stability(x1_val, x2_val)
        minimax_results.append(mm)
        
        logging.info(f"\n  Nominal values:")
        logging.info(f"    f1 = {mm['nominal_f1']:.6f}")
        logging.info(f"    f2 = {mm['nominal_f2']:.6f}")
        
        logging.info(f"\n  Worst-case increases (within ε={mm['epsilon']}):")
        logging.info(f"    Δf1 = {mm['worst_case_f1_increase']:.6f} "
              f"({mm['worst_case_f1_relative']*100:.2f}%)")
        logging.info(f"    Δf2 = {mm['worst_case_f2_increase']:.6f} "
              f"({mm['worst_case_f2_relative']*100:.2f}%)")
        
        logging.info(f"\n  Robustness Assessment:")
        logging.info(f"    Score: {mm['worst_case_score']:.6f} (lower is better)")
        logging.info(f"    Class: {mm['robustness_class']}")
    
    # Summary table
    logging.info("\n" + "=" * 70)
    logging.info("MINIMAX ROBUSTNESS SUMMARY")
    logging.info("=" * 70)
    logging.info(f"{'Sol':<5} {'x1':<10} {'x2':<10} {'Δf1(%)':<10} {'Δf2(%)':<10} "
          f"{'Score':<10} {'Robustness':<18}")
    logging.info("-" * 70)
    
    for i, ((x1, x2, f1, f2), mm) in enumerate(zip(pareto_solutions, minimax_results)):
        logging.info(f"{i+1:<5} {x1:<10.4f} {x2:<10.4f} "
              f"{mm['worst_case_f1_relative']*100:<10.2f} "
              f"{mm['worst_case_f2_relative']*100:<10.2f} "
              f"{mm['worst_case_score']:<10.4f} {mm['robustness_class']:<18}")
    
    logging.info("=" * 70)
    
    # Find most robust
    most_robust_idx = min(range(len(minimax_results)), 
                         key=lambda i: minimax_results[i]['worst_case_score'])
    logging.info(f"\nMost Robust Solution: #{most_robust_idx + 1}")
    logging.info(f"  x = ({pareto_solutions[most_robust_idx][0]:.6f}, "
          f"{pareto_solutions[most_robust_idx][1]:.6f})")
    logging.info(f"  Worst-case score: {minimax_results[most_robust_idx]['worst_case_score']:.6f}")
    logging.info(f"  Class: {minimax_results[most_robust_idx]['robustness_class']}")
    logging.info("=" * 70)
    
    # Plot results
    plot_results(pareto_solutions, minimax_results, timeout)
    
    solution_string = ' '.join(str(item) for item in pareto_solutions)
    return sha256(solution_string.encode()).hexdigest()

def plot_results(solutions, minimax_results, timeout):
    """Create clear visualizations for minimax stability analysis."""
    
    fig = plt.figure(figsize=(18, 10))
    
    x1_vals = [s[0] for s in solutions]
    x2_vals = [s[1] for s in solutions]
    f1_vals = [s[2] for s in solutions]
    f2_vals = [s[3] for s in solutions]
    colors = [mm['color'] for mm in minimax_results]
    
    # Plot 1: Decision Space with Worst-Case Perturbations
    ax1 = plt.subplot(2, 3, 1)
    
    x1_grid = np.linspace(0, 5, 300)
    x2_grid = np.linspace(0, 3, 300)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    
    C1 = (X1 - 5)**2 + X2**2
    C2 = (X1 - 8)**2 + (X2 + 3)**2
    feasible = (C1 <= 25) & (C2 >= 7.7)
    
    ax1.contourf(X1, X2, feasible.astype(int), levels=[0.5, 1.5], 
                 colors=['lightblue'], alpha=0.3)
    ax1.contour(X1, X2, C1, levels=[25], colors='blue', linewidths=2)
    ax1.contour(X1, X2, C2, levels=[7.7], colors='purple', linewidths=2)
    
    # Plot solutions with uncertainty circles
    for i, (x1, x2, mm) in enumerate(zip(x1_vals, x2_vals, minimax_results)):
        circle = plt.Circle((x1, x2), mm['epsilon'], color=colors[i], 
                           alpha=0.2, linewidth=0)
        ax1.add_patch(circle)
        
        # Draw arrows to worst-case points
        if mm['worst_direction_f1'] is not None:
            wp = mm['worst_point_f1']
            ax1.arrow(x1, x2, wp[0]-x1, wp[1]-x2, 
                     head_width=0.05, head_length=0.05, 
                     fc='red', ec='red', alpha=0.5, linewidth=1)
    
    ax1.scatter(x1_vals, x2_vals, c=colors, s=250, marker='o', 
                edgecolors='black', linewidths=3, zorder=5)
    
    for i in range(len(solutions)):
        ax1.annotate(f'{i+1}', (x1_vals[i], x2_vals[i]), 
                    ha='center', va='center', fontweight='bold', 
                    fontsize=12, color='white')
    
    ax1.set_xlabel('x₁', fontsize=13, fontweight='bold')
    ax1.set_ylabel('x₂', fontsize=13, fontweight='bold')
    ax1.set_title('Decision Space\n(circles show ε-neighborhood)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 3)
    ax1.set_aspect('equal')
    
    # Plot 2: Objective Space (Pareto Front)
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(f1_vals, f2_vals, c=colors, s=250, marker='o',
                edgecolors='black', linewidths=3, zorder=3)
    
    for i in range(len(solutions)):
        ax2.annotate(f'{i+1}', (f1_vals[i], f2_vals[i]), 
                    xytext=(8, 8), textcoords='offset points',
                    fontweight='bold', fontsize=11)
    
    ax2.set_xlabel('f₁(x) = 4x₁² + 4x₂²', fontsize=12)
    ax2.set_ylabel('f₂(x) = (x₁-5)² + (x₂-5)²', fontsize=12)
    ax2.set_title('Pareto Front\n(color = robustness)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Highly Robust'),
        Patch(facecolor='yellow', label='Moderately Robust'),
        Patch(facecolor='orange', label='Weakly Robust'),
        Patch(facecolor='red', label='Not Robust')
    ]
    ax2.legend(handles=legend_elements, loc='best', fontsize=10)
    
    # Plot 3: Worst-Case Score Bar Chart
    ax3 = plt.subplot(2, 3, 3)
    indices = list(range(1, len(solutions) + 1))
    scores = [mm['worst_case_score'] for mm in minimax_results]
    
    bars = ax3.bar(indices, scores, color=colors, edgecolor='black', linewidth=2)
    
    ax3.axhline(y=0.05, color='green', linestyle='--', alpha=0.6, 
                linewidth=2, label='Highly Robust threshold')
    ax3.axhline(y=0.15, color='orange', linestyle='--', alpha=0.6, 
                linewidth=2, label='Moderately Robust threshold')
    ax3.axhline(y=0.30, color='red', linestyle='--', alpha=0.6, 
                linewidth=2, label='Weakly Robust threshold')
    
    ax3.set_xlabel('Solution Number', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Worst-Case Score', fontsize=12, fontweight='bold')
    ax3.set_title('Minimax Robustness Scores\n(lower = more robust)', 
                  fontsize=14, fontweight='bold')
    ax3.set_xticks(indices)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Relative Degradation for f1
    ax4 = plt.subplot(2, 3, 4)
    f1_rel = [mm['worst_case_f1_relative'] * 100 for mm in minimax_results]
    
    bars = ax4.barh(indices, f1_rel, color=colors, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Solution Number', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Worst-Case f₁ Increase (%)', fontsize=12, fontweight='bold')
    ax4.set_title('f₁ Robustness\n(% degradation in worst-case)', 
                  fontsize=14, fontweight='bold')
    ax4.set_yticks(indices)
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.invert_yaxis()
    
    # Plot 5: Relative Degradation for f2
    ax5 = plt.subplot(2, 3, 5)
    f2_rel = [mm['worst_case_f2_relative'] * 100 for mm in minimax_results]
    
    bars = ax5.barh(indices, f2_rel, color=colors, edgecolor='black', linewidth=2)
    ax5.set_ylabel('Solution Number', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Worst-Case f₂ Increase (%)', fontsize=12, fontweight='bold')
    ax5.set_title('f₂ Robustness\n(% degradation in worst-case)', 
                  fontsize=14, fontweight='bold')
    ax5.set_yticks(indices)
    ax5.grid(True, alpha=0.3, axis='x')
    ax5.invert_yaxis()
    
    # Plot 6: Combined Degradation Comparison
    ax6 = plt.subplot(2, 3, 6)
    
    x = np.arange(len(solutions))
    width = 0.35
    
    ax6.bar(x - width/2, f1_rel, width, label='f₁ degradation', 
            color='steelblue', edgecolor='black', linewidth=1.5)
    ax6.bar(x + width/2, f2_rel, width, label='f₂ degradation', 
            color='coral', edgecolor='black', linewidth=1.5)
    
    ax6.set_xlabel('Solution Number', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Worst-Case Increase (%)', fontsize=12, fontweight='bold')
    ax6.set_title('Objective Degradation Comparison\n(worst-case scenario)', 
                  fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels([str(i+1) for i in range(len(solutions))])
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('BNH Problem: Minimax Stability Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if not inf == timeout:
        timer = fig.canvas.new_timer(interval=timeout, 
                                     callbacks=[(plt.close, [], {})])
        timer.start()
    
    plt.show()
    
    logging.info("\n✓ Visualization complete")

if __name__ == "__main__":
    rootpath = "." if len(argv) < 2 else argv[1]
    timeout = inf if len(argv) < 3 else float(argv[2])
    print(solve_bnh_z3_minimax(rootpath, timeout))

