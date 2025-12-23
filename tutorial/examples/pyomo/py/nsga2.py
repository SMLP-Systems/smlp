#!/usr/bin/python3.12
import pandas as pd
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from math import inf
from hashlib import sha256
from sys import argv
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create sample CSV files for demonstration
# In practice, you'd read these from actual CSV files

# Products data with multiple objectives
products_data = pd.DataFrame({
    'product': ['A', 'B', 'C', 'D'],
    'cost': [10, 15, 20, 12],
    'profit': [30, 45, 50, 35],
    'weight': [2, 3, 4, 2.5],
    'carbon_emission': [5, 8, 10, 6],  # Environmental impact
    'quality_score': [85, 90, 95, 80],  # Product quality
    'min_quantity': [5, 3, 2, 4],
    'max_quantity': [20, 25, 15, 30]
})

# Resources data
resources_data = pd.DataFrame({
    'resource': ['labor_hours', 'machine_hours', 'storage_space'],
    'capacity': [100, 80, 50]
})

# Product-Resource requirements
requirements_data = pd.DataFrame({
    'product': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'D', 'D', 'D'],
    'resource': ['labor_hours', 'machine_hours', 'storage_space'] * 4,
    'requirement': [2, 1, 0.5, 3, 2, 0.8, 4, 2.5, 1, 2.5, 1.5, 0.6]
})

# Create requirement dictionary for quick lookup
req_dict = {}
for _, row in requirements_data.iterrows():
    req_dict[(row['product'], row['resource'])] = row['requirement']

# ============================================================================
# CUSTOM CONSTRAINT FUNCTIONS (Python functions)
# ============================================================================

def calculate_resource_usage(quantities):
    """Calculate resource usage for each resource"""
    usage = []
    for idx, resource in enumerate(resources_data['resource']):
        capacity = resources_data.iloc[idx]['capacity']
        used = sum(req_dict.get((products_data.iloc[i]['product'], resource), 0) * quantities[i]
                  for i in range(len(quantities)))
        usage.append(used - capacity)  # Negative or zero means satisfied
    return usage

def calculate_weight_constraint(quantities):
    """Total weight constraint (max 60 units)"""
    total_weight = sum(products_data.iloc[i]['weight'] * quantities[i] 
                      for i in range(len(quantities)))
    return total_weight - 60  # Negative or zero means satisfied

def calculate_ratio_constraint(quantities):
    """Product B must be at least 30% of Product A"""
    product_a_qty = quantities[0]
    product_b_qty = quantities[1]
    return 0.3 * product_a_qty - product_b_qty  # Negative or zero means satisfied

def calculate_min_quantity_constraints(quantities):
    """Check minimum quantity constraints"""
    violations = []
    for i in range(len(quantities)):
        min_qty = products_data.iloc[i]['min_quantity']
        violations.append(min_qty - quantities[i])  # Negative or zero means satisfied
    return violations

# ============================================================================
# DEFINE MULTI-OBJECTIVE OPTIMIZATION PROBLEM
# ============================================================================

class ProductionProblem(Problem):
    """
    Multi-objective production planning problem
    
    Objectives:
    1. Maximize profit (converted to minimize negative profit)
    2. Minimize carbon emissions
    3. Maximize average quality (converted to minimize negative quality)
    
    Decision Variables: Quantity of each product to produce
    """
    
    def __init__(self):
        # Number of decision variables (one per product)
        n_var = len(products_data)
        
        # Number of objectives (profit, emissions, quality)
        n_obj = 3
        
        # Number of constraints
        # Resource constraints + weight constraint + ratio constraint + min quantity constraints
        n_constr = len(resources_data) + 1 + 1 + len(products_data)
        
        # Variable bounds (min and max quantity for each product)
        xl = products_data['min_quantity'].values
        xu = products_data['max_quantity'].values
        
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr,
                        xl=xl, xu=xu)
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate objectives and constraints for a population
        
        X: 2D array where each row is an individual (solution)
        out: Dictionary to store objectives and constraints
        """
        
        # Initialize arrays for objectives and constraints
        n_solutions = X.shape[0]
        f1 = np.zeros(n_solutions)  # Profit (to minimize negative)
        f2 = np.zeros(n_solutions)  # Carbon emissions (to minimize)
        f3 = np.zeros(n_solutions)  # Quality (to minimize negative)
        
        # Constraints array
        g = np.zeros((n_solutions, self.n_constr))
        
        # Evaluate each solution
        for i in range(n_solutions):
            quantities = X[i, :]
            
            # Objective 1: Maximize profit (minimize negative profit)
            profit = sum(products_data.iloc[j]['profit'] * quantities[j] 
                        for j in range(len(quantities)))
            f1[i] = -profit  # Negative because we minimize
            
            # Objective 2: Minimize carbon emissions
            emissions = sum(products_data.iloc[j]['carbon_emission'] * quantities[j] 
                           for j in range(len(quantities)))
            f2[i] = emissions
            
            # Objective 3: Maximize average quality (minimize negative quality)
            total_quantity = sum(quantities)
            if total_quantity > 0:
                avg_quality = sum(products_data.iloc[j]['quality_score'] * quantities[j] 
                                 for j in range(len(quantities))) / total_quantity
            else:
                avg_quality = 0
            f3[i] = -avg_quality  # Negative because we minimize
            
            # Constraints (g <= 0 means satisfied)
            constraint_idx = 0
            
            # Resource constraints
            resource_violations = calculate_resource_usage(quantities)
            for violation in resource_violations:
                g[i, constraint_idx] = violation
                constraint_idx += 1
            
            # Weight constraint
            g[i, constraint_idx] = calculate_weight_constraint(quantities)
            constraint_idx += 1
            
            # Ratio constraint
            g[i, constraint_idx] = calculate_ratio_constraint(quantities)
            constraint_idx += 1
            
            # Minimum quantity constraints
            min_qty_violations = calculate_min_quantity_constraints(quantities)
            for violation in min_qty_violations:
                g[i, constraint_idx] = violation
                constraint_idx += 1
        
        # Store objectives and constraints
        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = g

def main(rootpath: str = ".", timeout: float=5000) -> int:
    # Save to CSV
    products_data.to_csv(rootpath + '/products_nsga2.csv', index=False)
    resources_data.to_csv(rootpath + '/resources_nsga2.csv', index=False)
    requirements_data.to_csv(rootpath + '/requirements_nsga2.csv', index=False)
    
    print("=" * 80)
    print("NSGA-II MULTI-OBJECTIVE OPTIMIZATION with Pymoo")
    print("=" * 80)
    
    # ============================================================================
    # RUN NSGA-II ALGORITHM
    # ============================================================================
    
    print("\nSetting up NSGA-II algorithm...")
    
    # Create the problem instance
    problem = ProductionProblem()
    
    # Configure NSGA-II algorithm
    algorithm = NSGA2(
        pop_size=100,  # Population size
        sampling=FloatRandomSampling(),  # Random initial population
        crossover=SBX(prob=0.9, eta=15),  # Simulated Binary Crossover
        mutation=PM(eta=20),  # Polynomial Mutation
        eliminate_duplicates=True
    )
    
    # Set termination criterion
    termination = get_termination("n_gen", 100)  # Run for 100 generations
    
    print("Running NSGA-II optimization...")
    print(f"  Population Size: 100")
    print(f"  Generations: 100")
    print(f"  Objectives: 3 (Profit, Emissions, Quality)")
    print(f"  Constraints: {problem.n_constr}")
    print()
    
    # Run the optimization
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=True
    )
    
    # ============================================================================
    # ANALYZE RESULTS
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS - PARETO FRONT")
    print("=" * 80)
    
    # Get Pareto optimal solutions
    pareto_X = res.X  # Decision variables
    pareto_F = res.F  # Objective values
    
    print(f"\nNumber of Pareto optimal solutions found: {len(pareto_X)}")
    
    # Convert objectives back to original scale (profit and quality were negated)
    pareto_profits = -pareto_F[:, 0]
    pareto_emissions = pareto_F[:, 1]
    pareto_quality = -pareto_F[:, 2]
    
    # Create DataFrame with Pareto solutions
    pareto_df = pd.DataFrame({
        'Solution': range(1, len(pareto_X) + 1),
        'Profit': pareto_profits,
        'Carbon_Emissions': pareto_emissions,
        'Avg_Quality': pareto_quality
    })
    
    print("\n" + "-" * 80)
    print("Pareto Optimal Solutions (Sample - First 10):")
    print("-" * 80)
    print(pareto_df.head(10).to_string(index=False))
    
    # ============================================================================
    # ANALYZE SPECIFIC SOLUTIONS
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS OF KEY SOLUTIONS")
    print("=" * 80)
    
    # Find extreme solutions
    max_profit_idx = np.argmax(pareto_profits)
    min_emissions_idx = np.argmin(pareto_emissions)
    max_quality_idx = np.argmax(pareto_quality)
    
    solutions_to_analyze = [
        ("Maximum Profit", max_profit_idx),
        ("Minimum Emissions", min_emissions_idx),
        ("Maximum Quality", max_quality_idx)
    ]
    
    for name, idx in solutions_to_analyze:
        print(f"\n{'-' * 80}")
        print(f"{name} Solution:")
        print(f"{'-' * 80}")
        
        quantities = pareto_X[idx]
        
        print(f"Profit: ${pareto_profits[idx]:.2f}")
        print(f"Carbon Emissions: {pareto_emissions[idx]:.2f} units")
        print(f"Average Quality: {pareto_quality[idx]:.2f}")
        
        print("\nProduction Quantities:")
        prod_details = pd.DataFrame({
            'Product': products_data['product'],
            'Quantity': [round(q, 2) for q in quantities],
            'Profit': [round(q * products_data.iloc[i]['profit'], 2) 
                      for i, q in enumerate(quantities)],
            'Emissions': [round(q * products_data.iloc[i]['carbon_emission'], 2) 
                         for i, q in enumerate(quantities)]
        })
        print(prod_details.to_string(index=False))
        
        print("\nResource Utilization:")
        for res_idx, resource in enumerate(resources_data['resource']):
            capacity = resources_data.iloc[res_idx]['capacity']
            used = sum(req_dict.get((products_data.iloc[i]['product'], resource), 0) * quantities[i]
                      for i in range(len(quantities)))
            print(f"  {resource}: {used:.2f} / {capacity} ({used/capacity*100:.1f}%)")
    
    # ============================================================================
    # TRADE-OFF ANALYSIS
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("TRADE-OFF ANALYSIS")
    print("=" * 80)
    
    print(f"\nProfit Range: ${pareto_profits.min():.2f} - ${pareto_profits.max():.2f}")
    print(f"Emissions Range: {pareto_emissions.min():.2f} - {pareto_emissions.max():.2f} units")
    print(f"Quality Range: {pareto_quality.min():.2f} - {pareto_quality.max():.2f}")
    
    key_insights = "\nKey Insights:\n" + \
                  f"  • Maximizing profit results in {pareto_emissions[max_profit_idx]:.2f} emissions\n" + \
                  f"  • Minimizing emissions reduces profit to ${pareto_profits[min_emissions_idx]:.2f}\n" + \
                  f"  • High quality solution achieves {pareto_quality[max_quality_idx]:.2f} quality score"
    print(key_insights)
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("Use the Pareto front to select a solution based on your priorities:")
    print("  • High profit with acceptable environmental impact")
    print("  • Balance between all three objectives")
    print("  • Environmentally friendly with reasonable profit")
    print("\nExport pareto_df to CSV for further analysis or visualization.")
    print("=" * 80)
    
    # ============================================================================
    # VISUALIZE PARETO FRONT
    # ============================================================================
    
    
    print("\n" + "=" * 80)
    print("GENERATING PARETO FRONT VISUALIZATIONS")
    print("=" * 80)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 3D Pareto Front
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter = ax1.scatter(pareto_profits, pareto_emissions, pareto_quality, 
                         c=pareto_profits, cmap='viridis', s=50, alpha=0.6)
    ax1.set_xlabel('Profit ($)', fontsize=10)
    ax1.set_ylabel('Carbon Emissions', fontsize=10)
    ax1.set_zlabel('Avg Quality', fontsize=10)
    ax1.set_title('3D Pareto Front\n(All Three Objectives)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax1, label='Profit', shrink=0.5)
    
    # Highlight extreme solutions
    ax1.scatter([pareto_profits[max_profit_idx]], 
               [pareto_emissions[max_profit_idx]], 
               [pareto_quality[max_profit_idx]], 
               c='red', s=200, marker='*', edgecolors='black', linewidths=2,
               label='Max Profit')
    ax1.scatter([pareto_profits[min_emissions_idx]], 
               [pareto_emissions[min_emissions_idx]], 
               [pareto_quality[min_emissions_idx]], 
               c='green', s=200, marker='*', edgecolors='black', linewidths=2,
               label='Min Emissions')
    ax1.scatter([pareto_profits[max_quality_idx]], 
               [pareto_emissions[max_quality_idx]], 
               [pareto_quality[max_quality_idx]], 
               c='blue', s=200, marker='*', edgecolors='black', linewidths=2,
               label='Max Quality')
    ax1.legend(fontsize=8)
    
    # 2D: Profit vs Emissions
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(pareto_profits, pareto_emissions, alpha=0.6, s=50)
    ax2.scatter(pareto_profits[max_profit_idx], pareto_emissions[max_profit_idx], 
               c='red', s=200, marker='*', edgecolors='black', linewidths=2, label='Max Profit')
    ax2.scatter(pareto_profits[min_emissions_idx], pareto_emissions[min_emissions_idx], 
               c='green', s=200, marker='*', edgecolors='black', linewidths=2, label='Min Emissions')
    ax2.set_xlabel('Profit ($)', fontsize=10)
    ax2.set_ylabel('Carbon Emissions', fontsize=10)
    ax2.set_title('Profit vs Emissions Trade-off', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    
    # 2D: Profit vs Quality
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(pareto_profits, pareto_quality, alpha=0.6, s=50)
    ax3.scatter(pareto_profits[max_profit_idx], pareto_quality[max_profit_idx], 
               c='red', s=200, marker='*', edgecolors='black', linewidths=2, label='Max Profit')
    ax3.scatter(pareto_profits[max_quality_idx], pareto_quality[max_quality_idx], 
               c='blue', s=200, marker='*', edgecolors='black', linewidths=2, label='Max Quality')
    ax3.set_xlabel('Profit ($)', fontsize=10)
    ax3.set_ylabel('Avg Quality', fontsize=10)
    ax3.set_title('Profit vs Quality Trade-off', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    
    # 2D: Emissions vs Quality
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(pareto_emissions, pareto_quality, alpha=0.6, s=50)
    ax4.scatter(pareto_emissions[min_emissions_idx], pareto_quality[min_emissions_idx], 
               c='green', s=200, marker='*', edgecolors='black', linewidths=2, label='Min Emissions')
    ax4.scatter(pareto_emissions[max_quality_idx], pareto_quality[max_quality_idx], 
               c='blue', s=200, marker='*', edgecolors='black', linewidths=2, label='Max Quality')
    ax4.set_xlabel('Carbon Emissions', fontsize=10)
    ax4.set_ylabel('Avg Quality', fontsize=10)
    ax4.set_title('Emissions vs Quality Trade-off', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)
    
    # Parallel Coordinates Plot
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Normalize objectives to 0-1 range for better visualization
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    normalized_objectives = scaler.fit_transform(np.column_stack([pareto_profits, pareto_emissions, pareto_quality]))
    
    # Plot each solution as a line
    for i in range(len(normalized_objectives)):
        ax5.plot([0, 1, 2], normalized_objectives[i], alpha=0.3, linewidth=1)
    
    # Highlight extreme solutions
    ax5.plot([0, 1, 2], normalized_objectives[max_profit_idx], 
            c='red', linewidth=3, marker='o', markersize=8, label='Max Profit')
    ax5.plot([0, 1, 2], normalized_objectives[min_emissions_idx], 
            c='green', linewidth=3, marker='o', markersize=8, label='Min Emissions')
    ax5.plot([0, 1, 2], normalized_objectives[max_quality_idx], 
            c='blue', linewidth=3, marker='o', markersize=8, label='Max Quality')
    
    ax5.set_xticks([0, 1, 2])
    ax5.set_xticklabels(['Profit\n(normalized)', 'Emissions\n(normalized)', 'Quality\n(normalized)'], fontsize=9)
    ax5.set_ylabel('Normalized Value', fontsize=10)
    ax5.set_title('Parallel Coordinates Plot\n(All Solutions)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.legend(fontsize=8)
    
    # Distribution of objectives
    ax6 = fig.add_subplot(2, 3, 6)
    box_data = [pareto_profits, pareto_emissions, pareto_quality]
    bp = ax6.boxplot(box_data, tick_labels=['Profit', 'Emissions', 'Quality'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
        patch.set_facecolor(color)
    ax6.set_ylabel('Value', fontsize=10)
    ax6.set_title('Distribution of Objectives\nacross Pareto Front', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('pareto_front_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Pareto front visualizations saved as 'pareto_front_analysis.png'")
    
    # Create additional detailed plot for production quantities
    fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    solutions_info = [
        (max_profit_idx, 'Max Profit Solution', 'red'),
        (min_emissions_idx, 'Min Emissions Solution', 'green'),
        (max_quality_idx, 'Max Quality Solution', 'blue')
    ]
    
    for ax, (idx, title, color) in zip(axes, solutions_info):
        quantities = pareto_X[idx]
        products = products_data['product'].tolist()
        
        bars = ax.bar(products, quantities, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Product', fontsize=11)
        ax.set_ylabel('Quantity', fontsize=11)
        ax.set_title(f'{title}\nProfit: ${pareto_profits[idx]:.0f} | '
                    f'Emissions: {pareto_emissions[idx]:.0f} | '
                    f'Quality: {pareto_quality[idx]:.1f}', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('optimal_solutions_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Optimal solutions comparison saved as 'optimal_solutions_comparison.png'")
    if not inf == timeout:
        timer = fig.canvas.new_timer(interval=timeout, callbacks=[(plt.close, [], {})])
        timer.start()
    plt.show()
    
    print("\nVisualization complete! Two PNG files have been generated:")
    print("  1. pareto_front_analysis.png - Multi-view Pareto front analysis")
    print("  2. optimal_solutions_comparison.png - Detailed comparison of key solutions")
    print("=" * 80)
    return sha256(key_insights.encode()).hexdigest()

if __name__ == "__main__":
    rootpath = "." if len(argv) < 2 else argv[1]
    timeout = inf if len(argv) < 3 else argv[2]
    print(main(rootpath,timeout))
