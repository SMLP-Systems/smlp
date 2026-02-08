#!/usr/bin/python3.12
import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.variable import Real, Choice
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
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
    
    THREE OBJECTIVES (F1, F2, F3):
    Each category dominates in different regions of the 3D Pareto front
    by having different trade-off characteristics.
    """
    
    def __init__(self):
        # Define variables
        vars = {
            'x1': Real(bounds=(0, 10)),  # Float variable
            'x2': Choice(options=['A', 'B', 'C', 'D'])  # Categorical variable
        }
        
        super().__init__(vars=vars, n_obj=3, n_constr=0)
    
    def _evaluate(self, X, out, *args, **kwargs):
        # For mixed variables, X is a dictionary-like object
        x1 = np.array([x['x1'] for x in X])
        x2 = np.array([x['x2'] for x in X])
        
        F1 = np.zeros(len(x1))
        F2 = np.zeros(len(x1))
        F3 = np.zeros(len(x1))
        
        for i, cat in enumerate(x2):
            if cat == 'A':
                # Category A: Best for LOW F1, moderate F2, high F3
                F1[i] = 2 + 0.3 * x1[i]**2
                F2[i] = 20 + 0.5 * (x1[i] - 5)**2
                F3[i] = 40 - 2.5 * x1[i]
                
            elif cat == 'B':
                # Category B: Moderate F1, best for LOW F2, high F3
                F1[i] = 15 + 0.2 * (x1[i] - 3)**2
                F2[i] = 3 + 0.4 * x1[i]**2
                F3[i] = 35 - 2.0 * x1[i]
                
            elif cat == 'C':
                # Category C: High F1, moderate F2, best for LOW F3
                F1[i] = 30 - 1.5 * x1[i]
                F2[i] = 18 + 0.3 * (x1[i] - 6)**2
                F3[i] = 2 + 0.35 * x1[i]**2
                
            elif cat == 'D':
                # Category D: Moderate F1, high F2, balanced for LOW F3
                F1[i] = 20 + 0.25 * (x1[i] - 7)**2
                F2[i] = 32 - 1.8 * x1[i]
                F3[i] = 5 + 0.3 * (x1[i] - 4)**2
        
        out["F"] = np.column_stack([F1, F2, F3])


def main(rootpath: str = ".", timeout: float=5000) -> int:
    # Define the problem
    problem = MixedVariableProblem()
    
    # Create reference directions for NSGA-III
    # For 3 objectives, we use 'das-dennis' method
    # n_partitions controls the number of reference points
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=30)
    
    print(f"Number of reference directions: {len(ref_dirs)}")
    
    # Create the NSGA-III algorithm with mixed variable operators
    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=300,  # Can be auto-adjusted based on ref_dirs
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
                print(f"  X1={X1[idx]:6.3f}  →  F1={F[idx,0]:7.3f}, F2={F[idx,1]:7.3f}, F3={F[idx,2]:7.3f}")
            if count > 8:
                print(f"  ... and {count-8} more solutions")
    
    # Visualize Pareto front
    fig = plt.figure(figsize=(18, 10))
    
    # Plot 1: 3D Pareto front
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    colors = ['red', 'blue', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    
    for cat, color, marker in zip(categories, colors, markers):
        mask = X2 == cat
        if np.any(mask):
            ax1.scatter(F[mask, 0], F[mask, 1], F[mask, 2], 
                       c=color, marker=marker, label=f'Cat {cat}', 
                       s=80, alpha=0.7, edgecolors='black', linewidth=1)
    
    ax1.set_xlabel('F1', fontsize=11, fontweight='bold')
    ax1.set_ylabel('F2', fontsize=11, fontweight='bold')
    ax1.set_zlabel('F3', fontsize=11, fontweight='bold')
    ax1.set_title('3D Pareto Front', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: F1 vs F2 projection
    ax2 = plt.subplot(2, 3, 2)
    for cat, color, marker in zip(categories, colors, markers):
        mask = X2 == cat
        if np.any(mask):
            plt.scatter(F[mask, 0], F[mask, 1], c=color, marker=marker,
                       label=f'Cat {cat}', s=80, alpha=0.7, 
                       edgecolors='black', linewidth=1)
    
    plt.xlabel('F1', fontsize=11, fontweight='bold')
    plt.ylabel('F2', fontsize=11, fontweight='bold')
    plt.title('F1 vs F2 Projection', fontsize=13, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: F1 vs F3 projection
    ax3 = plt.subplot(2, 3, 3)
    for cat, color, marker in zip(categories, colors, markers):
        mask = X2 == cat
        if np.any(mask):
            plt.scatter(F[mask, 0], F[mask, 2], c=color, marker=marker,
                       label=f'Cat {cat}', s=80, alpha=0.7, 
                       edgecolors='black', linewidth=1)
    
    plt.xlabel('F1', fontsize=11, fontweight='bold')
    plt.ylabel('F3', fontsize=11, fontweight='bold')
    plt.title('F1 vs F3 Projection', fontsize=13, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Plot 4: F2 vs F3 projection
    ax4 = plt.subplot(2, 3, 4)
    for cat, color, marker in zip(categories, colors, markers):
        mask = X2 == cat
        if np.any(mask):
            plt.scatter(F[mask, 1], F[mask, 2], c=color, marker=marker,
                       label=f'Cat {cat}', s=80, alpha=0.7, 
                       edgecolors='black', linewidth=1)
    
    plt.xlabel('F2', fontsize=11, fontweight='bold')
    plt.ylabel('F3', fontsize=11, fontweight='bold')
    plt.title('F2 vs F3 Projection', fontsize=13, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Plot 5: X1 distribution by category
    ax5 = plt.subplot(2, 3, 5)
    
    for cat, color in zip(categories, colors):
        mask = X2 == cat
        if np.any(mask):
            plt.scatter([cat]*np.sum(mask), X1[mask], c=color, 
                       s=80, alpha=0.6, edgecolors='black', linewidth=1)
    
    plt.xlabel('Category', fontsize=11, fontweight='bold')
    plt.ylabel('X1 Values', fontsize=11, fontweight='bold')
    plt.title('X1 Distribution by Category', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.4, axis='y', linestyle='--')
    
    # Plot 6: Solution count by category
    ax6 = plt.subplot(2, 3, 6)
    category_counts = []
    
    for cat, color in zip(categories, colors):
        mask = X2 == cat
        count = np.sum(mask)
        category_counts.append(count)
        plt.bar(cat, count, color=color, alpha=0.8, edgecolor='black', linewidth=2)
        if count > 0:
            plt.text(cat, count + 0.5, str(count), ha='center', 
                    va='bottom', fontweight='bold', fontsize=11)
    
    plt.xlabel('Category', fontsize=11, fontweight='bold')
    plt.ylabel('Number of Solutions', fontsize=11, fontweight='bold')
    plt.title('Solutions per Category', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.4, axis='y', linestyle='--')
    
    plt.tight_layout()
    png_file=rootpath + '/nsga3_mixed_models_comparison.png'
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    print(f"✓ Optimal solutions comparison saved as {png_file}")
    if not inf == timeout:
        timer = fig.canvas.new_timer(interval=timeout, callbacks=[(plt.close, [], {})])
        timer.start()
    plt.show()
    with open(png_file, "rb") as image_file:
        image_data = image_file.read()
    return sha256(b64encode(image_data)).hexdigest()
    
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
            print(f"  F3  → min: {F[mask,2].min():6.2f}, max: {F[mask,2].max():6.2f}, " +
                  f"mean: {F[mask,2].mean():6.2f}")
        else:
            print(f"\nCategory {cat}: ⚠️  NO SOLUTIONS ⚠️")
    
    # Design explanation
    print("\n" + "="*70)
    print("PROBLEM DESIGN - 3-OBJECTIVE OPTIMIZATION")
    print("="*70)
    print("Category A: Best for LOW F1, moderate F2, high F3")
    print("Category B: Moderate F1, best for LOW F2, high F3")
    print("Category C: High F1, moderate F2, best for LOW F3")
    print("Category D: Moderate F1, high F2, balanced for LOW F3")
    print("\nEach category dominates in different regions of 3D objective space!")
    print("\n" + "="*70)
    print("NSGA-III ADVANTAGE FOR 3+ OBJECTIVES")
    print("="*70)
    print(f"Reference directions: {len(ref_dirs)}")
    print("NSGA-III excels at many-objective problems (3+ objectives)")
    print("by using reference points to maintain uniform diversity")
    print("across the entire Pareto surface, unlike NSGA-II which")
    print("struggles with crowding distance in high dimensions.")

if __name__ == "__main__":
    rootpath = "." if len(argv) < 2 else argv[1]
    timeout = inf if len(argv) < 3 else argv[2]
    print(main(rootpath,timeout))
