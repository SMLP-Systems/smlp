#!/usr/bin/python3.12
from z3 import *
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from math import inf
from hashlib import sha256
from base64 import b64encode

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
    
    weights = [(0.2, 0.8), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.8, 0.2)]
    
    for w1, w2 in weights:
        opt_w = Optimize()
        x1_w = Real(f'x1_w{w1}')
        x2_w = Real(f'x2_w{w1}')
        
        f1_w = 4 * x1_w * x1_w + 4 * x2_w * x2_w
        f2_w = (x1_w - 5) * (x1_w - 5) + (x2_w - 5) * (x2_w - 5)
        
        opt_w.add(x1_w >= 0, x1_w <= 5)
        opt_w.add(x2_w >= 0, x2_w <= 3)
        opt_w.add((x1_w - 5) * (x1_w - 5) + x2_w * x2_w <= 25)
        opt_w.add((x1_w - 8) * (x1_w - 8) + (x2_w + 3) * (x2_w + 3) >= 7.7)
        
        # Weighted objective (scaled for better numerical behavior)
        weighted_obj = w1 * f1_w / 100 + w2 * f2_w / 100
        opt_w.minimize(weighted_obj)
        
        if opt_w.check() == sat:
            model = opt_w.model()
            x1_val = float(model[x1_w].as_decimal(10).rstrip('?'))
            x2_val = float(model[x2_w].as_decimal(10).rstrip('?'))
            f1_val = 4 * x1_val**2 + 4 * x2_val**2
            f2_val = (x1_val - 5)**2 + (x2_val - 5)**2
            
            print(f"\nWeights (w1={w1}, w2={w2}):")
            print(f"  x1 = {x1_val:.6f}, x2 = {x2_val:.6f}")
            print(f"  f1 = {f1_val:.6f}, f2 = {f2_val:.6f}")
            pareto_solutions.append((x1_val, x2_val, f1_val, f2_val))
    
    # Verify constraints for all solutions
    print("\n4. Constraint Verification:")
    print("-" * 60)
    for i, (x1_val, x2_val, f1_val, f2_val) in enumerate(pareto_solutions):
        c1 = (x1_val - 5)**2 + x2_val**2
        c2 = (x1_val - 8)**2 + (x2_val + 3)**2
        print(f"Solution {i+1}:")
        print(f"  C1 = {c1:.6f} (should be ≤ 25): {'✓' if c1 <= 25 else '✗'}")
        print(f"  C2 = {c2:.6f} (should be ≥ 7.7): {'✓' if c2 >= 7.7 else '✗'}")
    
    # Plot Pareto front
    plot_results(pareto_solutions,timeout)

    solution_string=' '.join(str(item) for item in pareto_solutions)
    print(solution_string)
    return sha256(solution_string.encode()).hexdigest()

def plot_results(solutions,timeout):
    """Plot the Pareto front in objective space."""
    if not solutions:
        return
    
    f1_vals = [s[2] for s in solutions]
    f2_vals = [s[3] for s in solutions]
    
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(f1_vals, f2_vals, c='red', s=100, marker='o', 
                edgecolors='black', linewidths=2, zorder=3)
    
    for i, (x1, x2, f1, f2) in enumerate(solutions):
        plt.annotate(f'{i+1}', (f1, f2), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    plt.xlabel('f₁(x) = 4x₁² + 4x₂²', fontsize=12)
    plt.ylabel('f₂(x) = (x₁-5)² + (x₂-5)²', fontsize=12)
    plt.title('BNH Problem: Pareto Front Approximation', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if not inf == timeout:
        timer = fig.canvas.new_timer(interval=timeout, callbacks=[(plt.close, [], {})])
        timer.start()
    plt.show()
    print("\n5. Pareto Front Plot Generated")
    print("-" * 60)
    print("\n" + "=" * 60)
    print(f"Found {len(solutions)} Pareto optimal solutions")
    print("=" * 60)

if __name__ == "__main__":
    rootpath = "." if len(argv) < 2 else argv[1]
    timeout = inf if len(argv) < 3 else argv[2]
    print(solve_bnh_z3(rootpath,timeout))
