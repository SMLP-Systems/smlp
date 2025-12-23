#!/usr/bin/python3.12
import pandas as pd
import pyomo.environ as pyo
from hashlib import sha256
from pyomo.environ import *

# Create sample CSV files for demonstration
# In practice, you'd read these from actual CSV files

# Products data: cost, profit, weight per unit
products_data = pd.DataFrame({
    'product': ['A', 'B', 'C', 'D'],
    'cost': [10, 15, 20, 12],
    'profit': [30, 45, 50, 35],
    'weight': [2, 3, 4, 2.5],
    'min_quantity': [5, 3, 2, 4]
})

# Resources data: available capacity
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
#Create requirement parameter (product, resource) -> requirement
req_dict = {}
for _, row in requirements_data.iterrows():
    req_dict[(row['product'], row['resource'])] = row['requirement']
  

def main(rootpath: str = ".") -> str:
    # Save to CSV (optional - for demonstration)
    products_data.to_csv(rootpath + '/products.csv', index=False)
    resources_data.to_csv(rootpath + '/resources.csv', index=False)
    requirements_data.to_csv(rootpath + '/requirements.csv', index=False)
    
    print("=" * 60)
    print("OPTIMIZATION PROBLEM: Production Planning")
    print("=" * 60)
    
    # Create Pyomo model
    model = ConcreteModel()
    
    # Define sets from DataFrames
    model.products = Set(initialize=products_data['product'].tolist())
    model.resources = Set(initialize=resources_data['resource'].tolist())
    
    # Define parameters from DataFrames
    model.requirement = Param(model.products, model.resources, initialize=req_dict)
    model.cost = Param(model.products, initialize=products_data.set_index('product')['cost'].to_dict())
    model.profit = Param(model.products, initialize=products_data.set_index('product')['profit'].to_dict())
    model.weight = Param(model.products, initialize=products_data.set_index('product')['weight'].to_dict())
    model.min_qty = Param(model.products, initialize=products_data.set_index('product')['min_quantity'].to_dict())
    model.capacity = Param(model.resources, initialize=resources_data.set_index('resource')['capacity'].to_dict())
    
    # Decision variables: quantity of each product to produce
    model.quantity = Var(model.products, domain=NonNegativeReals)
    
    # Objective function: Maximize profit
    def objective_rule(m):
        return sum(m.profit[p] * m.quantity[p] for p in m.products)
    
    model.objective = Objective(rule=objective_rule, sense=maximize)
    
    # Standard Pyomo constraint: Resource capacity constraints
    def resource_constraint_rule(m, r):
        return sum(m.requirement[p, r] * m.quantity[p] for p in m.products) <= m.capacity[r]
    
    model.resource_constraint = Constraint(model.resources, rule=resource_constraint_rule)
    
    # CUSTOM PYTHON FUNCTION CONSTRAINTS
    
    # Constraint 1: Minimum quantity requirements (using Python function)
    def min_quantity_constraint(m, p):
        """Each product must meet minimum quantity - Python function constraint"""
        return m.quantity[p] >= m.min_qty[p]
    
    model.min_quantity_constraint = Constraint(model.products, rule=min_quantity_constraint)
    
    # Constraint 2: Complex business rule using Python function
    def complex_business_rule(m):
        """
        Complex constraint: Total weight cannot exceed 60 units AND
        product B quantity must be at least 30% of product A quantity
        This demonstrates a Python function with complex logic
        """
        total_weight = sum(m.weight[p] * m.quantity[p] for p in m.products)
        # Return a constraint expression
        return total_weight <= 60
    
    model.weight_limit = Constraint(rule=complex_business_rule)
    
    # Constraint 3: Ratio constraint using Python function
    def ratio_constraint(m):
        """Product B must be at least 30% of Product A"""
        return m.quantity['B'] >= 0.3 * m.quantity['A']
    
    model.ratio_constraint = Constraint(rule=ratio_constraint)
    
    # Constraint 4: Conditional constraint using Python function
    def conditional_constraint(m):
        """
        If producing more than 10 units of C, then D must be at least 8 units
        This is linearized for the solver
        """
        # For demonstration - in practice, you'd use indicator constraints
        # or big-M formulation for true if-then logic
        return m.quantity['D'] >= 0.5 * m.quantity['C']
    
    model.conditional_constraint = Constraint(rule=conditional_constraint)
    
    # Solve the model
    print("\nSolving optimization problem...")
    solver = SolverFactory('glpk')  # Using GLPK (free, open-source solver)
    results = solver.solve(model, tee=False)
    
    # Display results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    
    if results.solver.status == SolverStatus.ok:
        print(f"\nSolver Status: {results.solver.status}")
        print(f"Termination Condition: {results.solver.termination_condition}")
        optimal_objective_value=f"\nOptimal Objective Value (Max Profit): ${model.objective():.2f}"
        print(optimal_objective_value)
        
        print("\n" + "-" * 60)
        print("Production Quantities:")
        print("-" * 60)
        results_df = pd.DataFrame({
            'Product': list(model.products),
            'Quantity': [model.quantity[p].value for p in model.products],
            'Profit': [model.profit[p] for p in model.products],
            'Total_Profit': [model.quantity[p].value * model.profit[p] for p in model.products]
        })
        print(results_df.to_string(index=False))
        
        print("\n" + "-" * 60)
        print("Resource Utilization:")
        print("-" * 60)
        resource_usage = []
        for r in model.resources:
            used = sum(model.requirement[p, r] * model.quantity[p].value for p in model.products)
            resource_usage.append({
                'Resource': r,
                'Used': used,
                'Capacity': model.capacity[r],
                'Utilization_%': (used / model.capacity[r] * 100)
            })
        resource_df = pd.DataFrame(resource_usage)
        print(resource_df.to_string(index=False))
        
        print("\n" + "-" * 60)
        print("Constraint Verification:")
        print("-" * 60)
        total_weight = sum(model.weight[p] * model.quantity[p].value for p in model.products)
        print(f"Total Weight: {total_weight:.2f} (limit: 60)")
        print(f"Product B / Product A ratio: {model.quantity['B'].value / max(model.quantity['A'].value, 0.001):.2%} (min: 30%)")
        
    else:
        print(f"Solver Status: {results.solver.status}")
        error_message="Optimization failed!"
        print(error_message)
        return error_message
    
    print("\n" + "=" * 60)
    print("To use with your own CSV files:")
    print("  products_data = pd.read_csv('your_products.csv')")
    print("  resources_data = pd.read_csv('your_resources.csv')")
    print("  requirements_data = pd.read_csv('your_requirements.csv')")
    print("=" * 60)
    return sha256(optimal_objective_value.encode()).hexdigest()

if __name__ == "__main__":
    print(main())
