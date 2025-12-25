#!/usr/bin/python3.12
import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.variable import Real, Choice
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize
from matplotlib import pyplot as plt
from sys import argv
from math import inf
from base64 import b64encode
from hashlib import sha256

class MixedVariableProblem(Problem):
    """
    Multi-objective optimization with mixed variables:
    - X1: continuous float variable (0 to 10)
    - X2: categorical variable (A, B, C, D)
    
    Design principle: Each category dominates in different regions of the Pareto front
    by having different trade-off characteristics that intersect.
    """
    
    def __init__(self):
        # Define variables
        vars = {
            'x1': Real(bounds=(0, 10)),  # Float variable
            'x2': Choice(options=['A', 'B', 'C', 'D'])  # Categorical variable
        }
        
        super().__init__(vars=vars, n_obj=2, n_constr=0)
    
    def _evaluate(self, X, out, *args, **kwargs):
        # For mixed variables, X is a dictionary-like object
        x1 = np.array([x['x1'] for x in X])
        x2 = np.array([x['x2'] for x in X])
        
        F1 = np.zeros(len(x1))
        F2 = np.zeros(len(x1))
        
        for i, cat in enumerate(x2):
            if cat == 'A':
                # Category A: Best for LOW F1 (left side of Pareto front)
                # Gets worse as x1 increases
                F1[i] = 2 + 0.3 * x1[i]**2
                F2[i] = 40 - 2.5 * x1[i]
                
            elif cat == 'B':
                # Category B: Good for MID-LOW region
                # Competitive in the middle-left area
                F1[i] = 5 + 0.15 * (x1[i] - 3)**2
                F2[i] = 35 - 2.0 * x1[i]
                
            elif cat == 'C':
                # Category C: Good for MID-HIGH region  
                # Competitive in the middle-right area
                F1[i] = 7 + 0.15 * (x1[i] - 7)**2
                F2[i] = 30 - 1.5 * x1[i]
                
            elif cat == 'D':
                # Category D: Best for LOW F2 (right side of Pareto front)
                # Best at high x1 values
                F1[i] = 20 - 1.0 * x1[i]
                F2[i] = 2 + 0.3 * (x1[i] - 8)**2
        
        out["F"] = np.column_stack([F1, F2])


def main(rootpath: str = ".", timeout: float=5000):
    # Define the problem
    problem = MixedVariableProblem()
    
    # Create the NSGA-II algorithm with mixed variable operators
    algorithm = NSGA2(
        pop_size=200,  # Large population for diversity
        sampling=MixedVariableSampling(),
        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
        eliminate_duplicates=MixedVariableDuplicateElimination()
    )
    
    # Run optimization
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 250),  # Many generations for convergence
        seed=1,
        verbose=True
    )
    
    # Extract results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    
    # Pareto front solutions
    F = res.F  # Objective values
    X1 = np.array([x['x1'] for x in res.X])  # Float variable values
    X2 = np.array([x['x2'] for x in res.X])  # Categorical variable values
    
    print(f"\nTotal Pareto front solutions: {len(F)}")
    
    # Show solutions by category
    categories = ['A', 'B', 'C', 'D']
    print("\n" + "="*70)
    print("SOLUTIONS BY CATEGORY")
    print("="*70)
    
    for cat in categories:
        mask = X2 == cat
        count = np.sum(mask)
        print(f"\nCategory {cat}: {count} solutions")
        if count > 0:
            indices = np.where(mask)[0]
            # Sort by F1 for readability
            sorted_indices = indices[np.argsort(F[indices, 0])]
            for idx in sorted_indices[:8]:  # Show first 8 per category
                print(f"  X1={X1[idx]:6.3f}  →  F1={F[idx,0]:7.3f}, F2={F[idx,1]:7.3f}")
            if count > 8:
                print(f"  ... and {count-8} more solutions")
    
    # Visualize Pareto front
    fig = plt.figure(figsize=(18, 5))
    
    # Plot 1: Complete Pareto front
    ax1 = plt.subplot(1, 3, 1)
    colors = ['red', 'blue', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    
    for cat, color, marker in zip(categories, colors, markers):
        mask = X2 == cat
        if np.any(mask):
            # Sort by F1 for line connection
            sort_idx = np.argsort(F[mask, 0])
            f1_sorted = F[mask, 0][sort_idx]
            f2_sorted = F[mask, 1][sort_idx]
            plt.scatter(f1_sorted, f2_sorted, c=color, marker=marker,
                       label=f'Category {cat}', s=120, alpha=0.8, 
                       edgecolors='black', linewidth=1.5, zorder=3)
    
    plt.xlabel('F1', fontsize=13, fontweight='bold')
    plt.ylabel('F2', fontsize=13, fontweight='bold')
    plt.title('Pareto Front - All Categories', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.4, linestyle='--')
    
    # Plot 2: X1 distribution by category
    ax2 = plt.subplot(1, 3, 2)
    
    for cat, color in zip(categories, colors):
        mask = X2 == cat
        if np.any(mask):
            plt.scatter([cat]*np.sum(mask), X1[mask], c=color, 
                       s=100, alpha=0.6, edgecolors='black', linewidth=1.5)
    
    plt.xlabel('Category', fontsize=13, fontweight='bold')
    plt.ylabel('X1 Values', fontsize=13, fontweight='bold')
    plt.title('X1 Distribution by Category', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.4, axis='y', linestyle='--')
    
    # Plot 3: Solution count by category
    ax3 = plt.subplot(1, 3, 3)
    category_counts = []
    
    for cat, color in zip(categories, colors):
        mask = X2 == cat
        count = np.sum(mask)
        category_counts.append(count)
        plt.bar(cat, count, color=color, alpha=0.8, edgecolor='black', linewidth=2)
        if count > 0:
            plt.text(cat, count + 0.8, str(count), ha='center', 
                    va='bottom', fontweight='bold', fontsize=12)
    
    plt.xlabel('Category', fontsize=13, fontweight='bold')
    plt.ylabel('Number of Solutions', fontsize=13, fontweight='bold')
    plt.title('Solutions per Category', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.4, axis='y', linestyle='--')
    
    plt.tight_layout()
    png_file=rootpath + '/nsga2_mixed_models_comparison.png'
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    if not inf == timeout:
        timer = fig.canvas.new_timer(interval=timeout, callbacks=[(plt.close, [], {})])
        timer.start()

    # Summary statistics
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70)
    
    total = len(F)
    for cat in categories:
        mask = X2 == cat
        count = np.sum(mask)
        if count > 0:
            print(f"\nCategory {cat}: {count}/{total} solutions ({count/total*100:.1f}%)")
            print(f"  X1  → min: {X1[mask].min():5.2f}, max: {X1[mask].max():5.2f}, " +
                  f"mean: {X1[mask].mean():5.2f}, std: {X1[mask].std():5.2f}")
            print(f"  F1  → min: {F[mask,0].min():6.2f}, max: {F[mask,0].max():6.2f}, " +
                  f"mean: {F[mask,0].mean():6.2f}")
            print(f"  F2  → min: {F[mask,1].min():6.2f}, max: {F[mask,1].max():6.2f}, " +
                  f"mean: {F[mask,1].mean():6.2f}")
        else:
            print(f"\nCategory {cat}: ⚠️  NO SOLUTIONS ⚠️")
    
    # Design explanation
    print("\n" + "="*70)
    print("PROBLEM DESIGN - HOW CATEGORIES COMPETE")
    print("="*70)
    print("Category A: Dominates LEFT side (low F1, high F2)")
    print("Category B: Competitive in MID-LEFT region")
    print("Category C: Competitive in MID-RIGHT region")
    print("Category D: Dominates RIGHT side (high F1, low F2)")
    print("\nEach category is non-dominated in its specialized region!")

    plt.show()
    with open(png_file, "rb") as image_file:
        image_data = image_file.read()
    return sha256(b64encode(image_data)).hexdigest()

if __name__ == "__main__":
    rootpath = "." if len(argv) < 2 else argv[1]
    timeout = inf if len(argv) < 3 else argv[2]
    print(main(rootpath,timeout))
