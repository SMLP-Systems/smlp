#!/usr/bin/python3.12
import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms

# Create sample CSV files for demonstration
# In practice, you'd read these from actual CSV files

# Products data: cost, profit, weight per unit
products_data = pd.DataFrame({
    'product': ['A', 'B', 'C', 'D'],
    'cost': [10, 15, 20, 12],
    'profit': [30, 45, 50, 35],
    'weight': [2, 3, 4, 2.5],
    'min_quantity': [5, 3, 2, 4],
    'max_quantity': [20, 25, 15, 30]
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

# Save to CSV (optional - for demonstration)
products_data.to_csv('products_ga.csv', index=False)
resources_data.to_csv('resources_ga.csv', index=False)
requirements_data.to_csv('requirements_ga.csv', index=False)

print("=" * 70)
print("GENETIC ALGORITHM OPTIMIZATION: Production Planning with DEAP")
print("=" * 70)

# Create requirement dictionary for quick lookup
req_dict = {}
for _, row in requirements_data.iterrows():
    req_dict[(row['product'], row['resource'])] = row['requirement']

# ============================================================================
# CUSTOM CONSTRAINT FUNCTIONS (Python functions as requested)
# ============================================================================

def check_resource_constraints(individual):
    """Check if resource capacity constraints are satisfied"""
    for idx, resource in enumerate(resources_data['resource']):
        capacity = resources_data.iloc[idx]['capacity']
        used = 0
        for i, product in enumerate(products_data['product']):
            qty = individual[i]
            used += req_dict.get((product, resource), 0) * qty
        
        if used > capacity:
            return False
    return True

def check_min_quantity(individual):
    """Check minimum quantity constraints"""
    for i, row in products_data.iterrows():
        if individual[i] < row['min_quantity']:
            return False
    return True

def check_weight_limit(individual):
    """Complex constraint: Total weight cannot exceed 60 units"""
    total_weight = sum(products_data.iloc[i]['weight'] * individual[i] 
                      for i in range(len(individual)))
    return total_weight <= 60

def check_ratio_constraint(individual):
    """Product B must be at least 30% of Product A"""
    product_a_qty = individual[0]  # Product A
    product_b_qty = individual[1]  # Product B
    if product_a_qty > 0:
        return product_b_qty >= 0.3 * product_a_qty
    return True

def check_all_constraints(individual):
    """Apply all constraint functions"""
    return (check_resource_constraints(individual) and 
            check_min_quantity(individual) and 
            check_weight_limit(individual) and 
            check_ratio_constraint(individual))

# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def evaluate_profit(individual):
    """
    Objective function: Maximize profit
    Returns tuple for DEAP (must return tuple even for single objective)
    Applies penalty for constraint violations
    """
    # Calculate base profit
    total_profit = sum(products_data.iloc[i]['profit'] * individual[i] 
                      for i in range(len(individual)))
    
    # Apply penalty if constraints are violated
    if not check_all_constraints(individual):
        # Heavy penalty for constraint violation
        penalty = 10000
        
        # Calculate specific penalties for better guidance
        # Resource constraint penalties
        for idx, resource in enumerate(resources_data['resource']):
            capacity = resources_data.iloc[idx]['capacity']
            used = sum(req_dict.get((products_data.iloc[i]['product'], resource), 0) * individual[i]
                      for i in range(len(individual)))
            if used > capacity:
                penalty += (used - capacity) * 100
        
        # Weight constraint penalty
        total_weight = sum(products_data.iloc[i]['weight'] * individual[i] 
                          for i in range(len(individual)))
        if total_weight > 60:
            penalty += (total_weight - 60) * 50
        
        total_profit -= penalty
    
    return (total_profit,)  # Must return tuple

# ============================================================================
# DEAP SETUP
# ============================================================================

# Create fitness and individual classes
# FitnessMax means we're maximizing (use FitnessMin for minimization)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Define how to create genes (product quantities)
# Each gene is a float between min and max quantity for each product
def create_quantity(product_idx):
    min_qty = products_data.iloc[product_idx]['min_quantity']
    max_qty = products_data.iloc[product_idx]['max_quantity']
    return random.uniform(min_qty, max_qty)

# Register gene creation for each product
for i in range(len(products_data)):
    toolbox.register(f"attr_product_{i}", create_quantity, i)

# Create individual (chromosome) - list of quantities for all products
def create_individual():
    return creator.Individual([getattr(toolbox, f"attr_product_{i}")() 
                              for i in range(len(products_data))])

toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register genetic operators
toolbox.register("evaluate", evaluate_profit)
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2, indpb=0.2)  # Mutation
toolbox.register("select", tools.selTournament, tournsize=3)  # Selection

# Repair function to ensure quantities stay within bounds
def repair_individual(individual):
    for i in range(len(individual)):
        min_qty = products_data.iloc[i]['min_quantity']
        max_qty = products_data.iloc[i]['max_quantity']
        individual[i] = max(min_qty, min(max_qty, individual[i]))
    return individual

# ============================================================================
# RUN GENETIC ALGORITHM
# ============================================================================

def run_ga():
    random.seed(42)
    np.random.seed(42)
    
    # GA Parameters
    POPULATION_SIZE = 100
    GENERATIONS = 50
    CROSSOVER_PROB = 0.7
    MUTATION_PROB = 0.2
    
    print(f"\nGenetic Algorithm Parameters:")
    print(f"  Population Size: {POPULATION_SIZE}")
    print(f"  Generations: {GENERATIONS}")
    print(f"  Crossover Probability: {CROSSOVER_PROB}")
    print(f"  Mutation Probability: {MUTATION_PROB}")
    print("\nRunning optimization...\n")
    
    # Create initial population
    population = toolbox.population(n=POPULATION_SIZE)
    
    # Statistics tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Hall of Fame - keeps track of best individuals
    hof = tools.HallOfFame(1)
    
    # Run the algorithm
    population, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=CROSSOVER_PROB,
        mutpb=MUTATION_PROB,
        ngen=GENERATIONS,
        stats=stats,
        halloffame=hof,
        verbose=True
    )
    
    return population, logbook, hof

# Run the optimization
population, logbook, hof = run_ga()

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("OPTIMIZATION RESULTS")
print("=" * 70)

best_individual = hof[0]
best_individual = repair_individual(best_individual)

print(f"\nBest Profit: ${evaluate_profit(best_individual)[0]:.2f}")
print(f"Constraints Satisfied: {check_all_constraints(best_individual)}")

print("\n" + "-" * 70)
print("Optimal Production Quantities:")
print("-" * 70)
results_df = pd.DataFrame({
    'Product': products_data['product'],
    'Quantity': [round(qty, 2) for qty in best_individual],
    'Profit_per_Unit': products_data['profit'],
    'Total_Profit': [round(qty * products_data.iloc[i]['profit'], 2) 
                     for i, qty in enumerate(best_individual)]
})
print(results_df.to_string(index=False))

print("\n" + "-" * 70)
print("Resource Utilization:")
print("-" * 70)
resource_usage = []
for idx, resource in enumerate(resources_data['resource']):
    capacity = resources_data.iloc[idx]['capacity']
    used = sum(req_dict.get((products_data.iloc[i]['product'], resource), 0) * best_individual[i]
              for i in range(len(best_individual)))
    resource_usage.append({
        'Resource': resource,
        'Used': round(used, 2),
        'Capacity': capacity,
        'Utilization_%': round(used / capacity * 100, 2)
    })
resource_df = pd.DataFrame(resource_usage)
print(resource_df.to_string(index=False))

print("\n" + "-" * 70)
print("Constraint Verification:")
print("-" * 70)
total_weight = sum(products_data.iloc[i]['weight'] * best_individual[i] 
                  for i in range(len(best_individual)))
print(f"Total Weight: {total_weight:.2f} (limit: 60)")
ratio = best_individual[1] / max(best_individual[0], 0.001)
print(f"Product B / Product A ratio: {ratio:.2%} (min: 30%)")
print(f"All resource constraints: {'✓ Satisfied' if check_resource_constraints(best_individual) else '✗ Violated'}")
print(f"Minimum quantities: {'✓ Satisfied' if check_min_quantity(best_individual) else '✗ Violated'}")

print("\n" + "-" * 70)
print("Evolution Statistics:")
print("-" * 70)
gen = logbook.select("gen")
max_fitness = logbook.select("max")
avg_fitness = logbook.select("avg")

print(f"Generation 0 - Max Fitness: ${max_fitness[0]:.2f}, Avg: ${avg_fitness[0]:.2f}")
print(f"Generation {len(gen)//2} - Max Fitness: ${max_fitness[len(gen)//2]:.2f}, Avg: ${avg_fitness[len(gen)//2]:.2f}")
print(f"Generation {len(gen)-1} - Max Fitness: ${max_fitness[-1]:.2f}, Avg: ${avg_fitness[-1]:.2f}")

print("\n" + "=" * 70)
print("To use with your own CSV files:")
print("  products_data = pd.read_csv('your_products.csv')")
print("  resources_data = pd.read_csv('your_resources.csv')")
print("  requirements_data = pd.read_csv('your_requirements.csv')")
print("=" * 70)
