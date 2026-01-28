#!/usr/bin/python3.12
from pysmt.shortcuts import Symbol, Real, Solver, And, Plus, Times, Minus, LE, GE
from pysmt.typing import REAL
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sys import argv
from math import inf
from hashlib import sha256
from base64 import b64encode
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


def solve_bnh_pysmt(rootpath: str = ".", timeout: float=5000) -> str:
    """
    Solve the BNH multi-objective optimization problem using PySMT with Z3.
    Uses multiple approaches to find Pareto optimal solutions.
    """
    
    logging.info("=" * 60)
    logging.info("BNH Multi-Objective Optimization Problem with PySMT")
    logging.info("=" * 60)
    
    # Store Pareto solutions with method labels
    pareto_solutions = []
    
    # Approach 1: Minimize f1 with constraint satisfaction
    logging.info("\n1. Optimization: Minimize f1")
    logging.info("-" * 60)
    
    result = minimize_objective_1()
    if result:
        pareto_solutions.append((*result, 'minimize_f1'))
        x1_val, x2_val, f1_val, f2_val = result
        logging.info(f"x1 = {x1_val:.6f}")
        logging.info(f"x2 = {x2_val:.6f}")
        logging.info(f"f1 = {f1_val:.6f}")
        logging.info(f"f2 = {f2_val:.6f}")
    
    # Approach 2: Minimize f2 with constraint satisfaction
    logging.info("\n2. Optimization: Minimize f2")
    logging.info("-" * 60)
    
    result = minimize_objective_2()
    if result:
        pareto_solutions.append((*result, 'minimize_f2'))
        x1_val, x2_val, f1_val, f2_val = result
        logging.info(f"x1 = {x1_val:.6f}")
        logging.info(f"x2 = {x2_val:.6f}")
        logging.info(f"f1 = {f1_val:.6f}")
        logging.info(f"f2 = {f2_val:.6f}")
    
    # Approach 3: Weighted sum method with different weights
    logging.info("\n3. Weighted Sum Method (various weights):")
    logging.info("-" * 60)
    
    weights = [(0.2, 0.8), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.8, 0.2)]
    
    for w1, w2 in weights:
        result = minimize_weighted(w1, w2)
        if result:
            x1_val, x2_val, f1_val, f2_val = result
            logging.info(f"\nWeights (w1={w1}, w2={w2}):")
            logging.info(f"  x1 = {x1_val:.6f}, x2 = {x2_val:.6f}")
            logging.info(f"  f1 = {f1_val:.6f}, f2 = {f2_val:.6f}")
            pareto_solutions.append((*result, 'weighted'))
    
    # Approach 4: Epsilon Constraint Method
    logging.info("\n4. Epsilon Constraint Method (Pareto Front Simulation):")
    logging.info("-" * 60)
    
    epsilon_solutions = epsilon_constraint_method(num_points=10)
    for result in epsilon_solutions:
        if result:
            x1_val, x2_val, f1_val, f2_val = result
            logging.info(f"  x1 = {x1_val:.6f}, x2 = {x2_val:.6f}")
            logging.info(f"  f1 = {f1_val:.6f}, f2 = {f2_val:.6f}")
            pareto_solutions.append((*result, 'epsilon'))
    
    # Verify constraints for all solutions
    logging.info("\n5. Constraint Verification:")
    logging.info("-" * 60)
    for i, solution in enumerate(pareto_solutions):
        x1_val, x2_val, f1_val, f2_val, method = solution
        c1 = (x1_val - 5)**2 + x2_val**2
        c2 = (x1_val - 8)**2 + (x2_val + 3)**2
        logging.info(f"Solution {i+1} ({method}):")
        logging.info(f"  C1 = {c1:.6f} (should be ≤ 25): {'✓' if c1 <= 25 else '✗'}")
        logging.info(f"  C2 = {c2:.6f} (should be ≥ 7.7): {'✓' if c2 >= 7.7 else '✗'}")
    
    # Minimax Stability Analysis
    logging.info("\n6. Minimax Stability Analysis:")
    logging.info("=" * 60)
    logging.info("Minimax Philosophy:")
    logging.info("  - Optimizes for WORST-CASE scenario")
    logging.info("  - Ensures robustness under uncertainty")
    logging.info("  - Provides strong guarantees against adversarial perturbations")
    logging.info("=" * 60)
    
    minimax_results = []
    epsilon_uncertainty = 0.001  # Uncertainty radius
    
    for i, solution in enumerate(pareto_solutions):
        x1_val, x2_val, f1_val, f2_val, method = solution
        logging.info(f"\nSolution {i+1} ({method}): x = ({x1_val:.6f}, {x2_val:.6f})")
        logging.info(f"  Nominal: f1 = {f1_val:.6f}, f2 = {f2_val:.6f}")
        
        mm = check_minimax_stability(x1_val, x2_val, epsilon=epsilon_uncertainty)
        minimax_results.append(mm)
        
        logging.info(f"\n  Worst-Case Degradation (ε = {epsilon_uncertainty}):")
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
    logging.info(f"{'Sol':<5} {'Method':<18} {'x1':<10} {'x2':<10} {'Δf1(%)':<10} "
          f"{'Δf2(%)':<10} {'Score':<10} {'Robustness':<18}")
    logging.info("-" * 70)
    
    for i, (solution, mm) in enumerate(zip(pareto_solutions, minimax_results)):
        x1_val, x2_val, f1_val, f2_val, method = solution
        logging.info(f"{i+1:<5} {method:<18} {x1_val:<10.4f} {x2_val:<10.4f} "
              f"{mm['worst_case_f1_relative']*100:<10.2f} "
              f"{mm['worst_case_f2_relative']*100:<10.2f} "
              f"{mm['worst_case_score']:<10.4f} {mm['robustness_class']:<18}")
    
    logging.info("=" * 70)
    
    # Find most robust
    most_robust_idx = min(range(len(minimax_results)), 
                         key=lambda i: minimax_results[i]['worst_case_score'])
    robust_solution = pareto_solutions[most_robust_idx]
    logging.info(f"\nMost Robust Solution: #{most_robust_idx + 1}")
    logging.info(f"  Method: {robust_solution[4]}")
    logging.info(f"  x = ({robust_solution[0]:.6f}, {robust_solution[1]:.6f})")
    logging.info(f"  Worst-case score: {minimax_results[most_robust_idx]['worst_case_score']:.6f}")
    logging.info(f"  Class: {minimax_results[most_robust_idx]['robustness_class']}")
    logging.info("=" * 70)
    
    # Plot Pareto front
    plot_results(pareto_solutions, minimax_results, timeout)

    solution_string = ' '.join(str(item[:4]) for item in pareto_solutions)
    logging.info(solution_string)
    return sha256(solution_string.encode()).hexdigest()


def get_constraints(x1, x2):
    """Define the constraints for the BNH problem using PySMT."""
    constraints = []
    
    # Variable bounds
    constraints.append(GE(x1, Real(0)))
    constraints.append(LE(x1, Real(5)))
    constraints.append(GE(x2, Real(0)))
    constraints.append(LE(x2, Real(3)))
    
    # C1: (x1 - 5)^2 + x2^2 <= 25
    x1_minus_5 = Plus(x1, Real(-5))
    term1 = Times(x1_minus_5, x1_minus_5)
    term2 = Times(x2, x2)
    c1_lhs = Plus(term1, term2)
    constraints.append(LE(c1_lhs, Real(25)))
    
    # C2: (x1 - 8)^2 + (x2 + 3)^2 >= 7.7
    x1_minus_8 = Plus(x1, Real(-8))
    x2_plus_3 = Plus(x2, Real(3))
    term3 = Times(x1_minus_8, x1_minus_8)
    term4 = Times(x2_plus_3, x2_plus_3)
    c2_lhs = Plus(term3, term4)
    constraints.append(GE(c2_lhs, Real(7.7)))
    
    return And(constraints)


def evaluate_objectives(x1_val, x2_val):
    """Evaluate both objective functions."""
    f1 = 4 * x1_val**2 + 4 * x2_val**2
    f2 = (x1_val - 5)**2 + (x2_val - 5)**2
    return f1, f2


def minimize_objective_1():
    """Minimize f1 = 4*x1^2 + 4*x2^2 using binary search."""
    x1 = Symbol("x1", REAL)
    x2 = Symbol("x2", REAL)
    
    constraints = get_constraints(x1, x2)
    
    # Binary search on f1
    min_bound = 0.0
    max_bound = 400.0  # Upper bound for f1
    epsilon = 0.01
    best_solution = None
    
    iteration = 0
    max_iterations = 50
    
    while max_bound - min_bound > epsilon and iteration < max_iterations:
        iteration += 1
        mid = (min_bound + max_bound) / 2
        
        # f1 = 4*x1^2 + 4*x2^2 <= mid
        x1_squared = Times(x1, x1)
        x2_squared = Times(x2, x2)
        f1_expr = Plus(Times(Real(4), x1_squared), Times(Real(4), x2_squared))
        obj_constraint = LE(f1_expr, Real(mid))
        
        full_constraint = And(constraints, obj_constraint)
        
        with Solver(name="z3") as solver:
            solver.add_assertion(full_constraint)
            
            if solver.solve():
                model = solver.get_model()
                x1_val = float(model.get_value(x1).constant_value())
                x2_val = float(model.get_value(x2).constant_value())
                f1_val, f2_val = evaluate_objectives(x1_val, x2_val)
                
                best_solution = (x1_val, x2_val, f1_val, f2_val)
                max_bound = mid
            else:
                min_bound = mid
    
    return best_solution


def minimize_objective_2():
    """Minimize f2 = (x1-5)^2 + (x2-5)^2 using binary search."""
    x1 = Symbol("x1_f2", REAL)
    x2 = Symbol("x2_f2", REAL)
    
    constraints = get_constraints(x1, x2)
    
    # Binary search on f2
    min_bound = 0.0
    max_bound = 100.0  # Upper bound for f2
    epsilon = 0.01
    best_solution = None
    
    iteration = 0
    max_iterations = 50
    
    while max_bound - min_bound > epsilon and iteration < max_iterations:
        iteration += 1
        mid = (min_bound + max_bound) / 2
        
        # f2 = (x1-5)^2 + (x2-5)^2 <= mid
        x1_minus_5 = Plus(x1, Real(-5))
        x2_minus_5 = Plus(x2, Real(-5))
        term1 = Times(x1_minus_5, x1_minus_5)
        term2 = Times(x2_minus_5, x2_minus_5)
        f2_expr = Plus(term1, term2)
        obj_constraint = LE(f2_expr, Real(mid))
        
        full_constraint = And(constraints, obj_constraint)
        
        with Solver(name="z3") as solver:
            solver.add_assertion(full_constraint)
            
            if solver.solve():
                model = solver.get_model()
                x1_val = float(model.get_value(x1).constant_value())
                x2_val = float(model.get_value(x2).constant_value())
                f1_val, f2_val = evaluate_objectives(x1_val, x2_val)
                
                best_solution = (x1_val, x2_val, f1_val, f2_val)
                max_bound = mid
            else:
                min_bound = mid
    
    return best_solution


def minimize_weighted(w1, w2):
    """Minimize weighted sum: w1*f1 + w2*f2 using binary search."""
    x1 = Symbol(f"x1_w{w1}", REAL)
    x2 = Symbol(f"x2_w{w1}", REAL)
    
    constraints = get_constraints(x1, x2)
    
    # Binary search on weighted objective
    min_bound = 0.0
    max_bound = 500.0  # Upper bound for weighted sum
    epsilon = 0.01
    best_solution = None
    
    iteration = 0
    max_iterations = 50
    
    while max_bound - min_bound > epsilon and iteration < max_iterations:
        iteration += 1
        mid = (min_bound + max_bound) / 2
        
        # f1 = 4*x1^2 + 4*x2^2
        x1_squared = Times(x1, x1)
        x2_squared = Times(x2, x2)
        f1_expr = Plus(Times(Real(4), x1_squared), Times(Real(4), x2_squared))
        
        # f2 = (x1-5)^2 + (x2-5)^2
        x1_minus_5 = Plus(x1, Real(-5))
        x2_minus_5 = Plus(x2, Real(-5))
        term1 = Times(x1_minus_5, x1_minus_5)
        term2 = Times(x2_minus_5, x2_minus_5)
        f2_expr = Plus(term1, term2)
        
        # Weighted objective: w1*f1 + w2*f2 <= mid
        weighted = Plus(Times(Real(w1), f1_expr), Times(Real(w2), f2_expr))
        obj_constraint = LE(weighted, Real(mid))
        
        full_constraint = And(constraints, obj_constraint)
        
        with Solver(name="z3") as solver:
            solver.add_assertion(full_constraint)
            
            if solver.solve():
                model = solver.get_model()
                x1_val = float(model.get_value(x1).constant_value())
                x2_val = float(model.get_value(x2).constant_value())
                f1_val, f2_val = evaluate_objectives(x1_val, x2_val)
                
                best_solution = (x1_val, x2_val, f1_val, f2_val)
                max_bound = mid
            else:
                min_bound = mid
    
    return best_solution


def check_minimax_stability(x1_val, x2_val, epsilon=0.001, num_directions=100):
    """
    Minimax stability check: finds worst-case performance degradation.
    Philosophy: Optimize for worst-case scenario to ensure robustness.
    
    Args:
        x1_val: x1 coordinate of solution
        x2_val: x2 coordinate of solution
        epsilon: radius of uncertainty ball
        num_directions: number of directions to search
        
    Returns:
        Dictionary with robustness metrics
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


def epsilon_constraint_method(num_points=10):
    """
    Generate Pareto front using epsilon constraint method.
    
    Strategy: Minimize f1 while constraining f2 to different epsilon values.
    This systematically explores the trade-off between objectives.
    
    Args:
        num_points: Number of points to generate along the Pareto front
        
    Returns:
        List of solutions (x1, x2, f1, f2)
    """
    solutions = []
    
    # First, find the range of f2 values
    # Get minimum f2 (optimizing f2 alone)
    x1 = Symbol("x1_eps_range", REAL)
    x2 = Symbol("x2_eps_range", REAL)
    constraints = get_constraints(x1, x2)
    
    # Find minimum f2
    min_bound = 0.0
    max_bound = 100.0
    epsilon = 0.01
    min_f2 = None
    
    iteration = 0
    max_iterations = 50
    
    while max_bound - min_bound > epsilon and iteration < max_iterations:
        iteration += 1
        mid = (min_bound + max_bound) / 2
        
        x1_minus_5 = Plus(x1, Real(-5))
        x2_minus_5 = Plus(x2, Real(-5))
        term1 = Times(x1_minus_5, x1_minus_5)
        term2 = Times(x2_minus_5, x2_minus_5)
        f2_expr = Plus(term1, term2)
        obj_constraint = LE(f2_expr, Real(mid))
        
        full_constraint = And(constraints, obj_constraint)
        
        with Solver(name="z3") as solver:
            solver.add_assertion(full_constraint)
            
            if solver.solve():
                model = solver.get_model()
                x1_val = float(model.get_value(x1).constant_value())
                x2_val = float(model.get_value(x2).constant_value())
                _, f2_val = evaluate_objectives(x1_val, x2_val)
                min_f2 = f2_val
                max_bound = mid
            else:
                min_bound = mid
    
    # Find maximum f2 (when minimizing f1)
    x1 = Symbol("x1_eps_max", REAL)
    x2 = Symbol("x2_eps_max", REAL)
    constraints = get_constraints(x1, x2)
    
    min_bound = 0.0
    max_bound = 400.0
    epsilon = 0.01
    max_f2 = None
    
    iteration = 0
    max_iterations = 50
    
    while max_bound - min_bound > epsilon and iteration < max_iterations:
        iteration += 1
        mid = (min_bound + max_bound) / 2
        
        x1_squared = Times(x1, x1)
        x2_squared = Times(x2, x2)
        f1_expr = Plus(Times(Real(4), x1_squared), Times(Real(4), x2_squared))
        obj_constraint = LE(f1_expr, Real(mid))
        
        full_constraint = And(constraints, obj_constraint)
        
        with Solver(name="z3") as solver:
            solver.add_assertion(full_constraint)
            
            if solver.solve():
                model = solver.get_model()
                x1_val = float(model.get_value(x1).constant_value())
                x2_val = float(model.get_value(x2).constant_value())
                _, f2_val = evaluate_objectives(x1_val, x2_val)
                max_f2 = f2_val
                max_bound = mid
            else:
                min_bound = mid
    
    if min_f2 is None or max_f2 is None:
        logging.info("  Could not determine f2 range")
        return solutions
    
    logging.info(f"  f2 range: [{min_f2:.6f}, {max_f2:.6f}]")
    
    # Generate epsilon values
    epsilon_values = np.linspace(min_f2, max_f2, num_points)
    
    # For each epsilon value, minimize f1 subject to f2 <= epsilon
    for eps_val in epsilon_values:
        # Convert numpy float to Python float for PySMT
        eps_val = float(eps_val)
        x1 = Symbol(f"x1_eps_{eps_val}", REAL)
        x2 = Symbol(f"x2_eps_{eps_val}", REAL)
        
        constraints = get_constraints(x1, x2)
        
        # Add constraint: f2 <= epsilon
        x1_minus_5 = Plus(x1, Real(-5))
        x2_minus_5 = Plus(x2, Real(-5))
        term1 = Times(x1_minus_5, x1_minus_5)
        term2 = Times(x2_minus_5, x2_minus_5)
        f2_expr = Plus(term1, term2)
        f2_constraint = LE(f2_expr, Real(eps_val))
        
        constraints_with_epsilon = And(constraints, f2_constraint)
        
        # Binary search to minimize f1
        min_bound = 0.0
        max_bound = 400.0
        epsilon = 0.01
        best_solution = None
        
        iteration = 0
        max_iterations = 50
        
        while max_bound - min_bound > epsilon and iteration < max_iterations:
            iteration += 1
            mid = (min_bound + max_bound) / 2
            
            # f1 = 4*x1^2 + 4*x2^2 <= mid
            x1_squared = Times(x1, x1)
            x2_squared = Times(x2, x2)
            f1_expr = Plus(Times(Real(4), x1_squared), Times(Real(4), x2_squared))
            f1_constraint = LE(f1_expr, Real(mid))
            
            full_constraint = And(constraints_with_epsilon, f1_constraint)
            
            with Solver(name="z3") as solver:
                solver.add_assertion(full_constraint)
                
                if solver.solve():
                    model = solver.get_model()
                    x1_val = float(model.get_value(x1).constant_value())
                    x2_val = float(model.get_value(x2).constant_value())
                    f1_val, f2_val = evaluate_objectives(x1_val, x2_val)
                    
                    best_solution = (x1_val, x2_val, f1_val, f2_val)
                    max_bound = mid
                else:
                    min_bound = mid
        
        if best_solution:
            solutions.append(best_solution)
    
    return solutions


def plot_results(solutions, minimax_results, timeout):
    """Plot the Pareto front with minimax stability analysis visualization."""
    if not solutions:
        return
    
    # Separate solutions by method
    methods = {}
    for i, solution in enumerate(solutions):
        x1, x2, f1, f2, method = solution
        if method not in methods:
            methods[method] = {'f1': [], 'f2': [], 'solutions': [], 'indices': []}
        methods[method]['f1'].append(f1)
        methods[method]['f2'].append(f2)
        methods[method]['solutions'].append((x1, x2, f1, f2))
        methods[method]['indices'].append(i)
    
    # Define colors and labels for each method
    method_styles = {
        'minimize_f1': {'color': 'blue', 'marker': 's', 'label': 'Minimize f₁', 'size': 120},
        'minimize_f2': {'color': 'green', 'marker': 's', 'label': 'Minimize f₂', 'size': 120},
        'weighted': {'color': 'red', 'marker': 'o', 'label': 'Weighted Sum', 'size': 100},
        'epsilon': {'color': 'purple', 'marker': '^', 'label': 'Epsilon Constraint', 'size': 100}
    }
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 10))
    
    x1_vals = [s[0] for s in solutions]
    x2_vals = [s[1] for s in solutions]
    f1_vals = [s[2] for s in solutions]
    f2_vals = [s[3] for s in solutions]
    robustness_colors = [mm['color'] for mm in minimax_results]
    
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
    ax1.contour(X1, X2, C1, levels=[25], colors='blue', linewidths=2, label='C1 boundary')
    ax1.contour(X1, X2, C2, levels=[7.7], colors='purple', linewidths=2, label='C2 boundary')
    
    # Plot solutions with uncertainty circles
    for i, (x1, x2, mm) in enumerate(zip(x1_vals, x2_vals, minimax_results)):
        circle = plt.Circle((x1, x2), mm['epsilon'], color=robustness_colors[i], 
                           alpha=0.2, linewidth=0)
        ax1.add_patch(circle)
        
        # Draw arrows to worst-case points
        if mm['worst_direction_f1'] is not None:
            wp = mm['worst_point_f1']
            ax1.arrow(x1, x2, wp[0]-x1, wp[1]-x2, 
                     head_width=0.05, head_length=0.05, 
                     fc='red', ec='red', alpha=0.5, linewidth=1)
    
    ax1.scatter(x1_vals, x2_vals, c=robustness_colors, s=250, marker='o', 
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
    
    # Plot 2: Objective Space (Pareto Front) colored by method
    ax2 = plt.subplot(2, 3, 2)
    
    # Plot each method with its own color and marker
    for method, data in methods.items():
        style = method_styles.get(method, {'color': 'gray', 'marker': 'o', 'label': method, 'size': 100})
        ax2.scatter(data['f1'], data['f2'], 
                   c=style['color'], 
                   s=style['size'], 
                   marker=style['marker'],
                   edgecolors='black', 
                   linewidths=1.5, 
                   label=style['label'],
                   alpha=0.8,
                   zorder=3)
    
    for i in range(len(solutions)):
        ax2.annotate(f'{i+1}', (f1_vals[i], f2_vals[i]), 
                    xytext=(8, 8), textcoords='offset points',
                    fontweight='bold', fontsize=11)
    
    ax2.set_xlabel('f₁(x) = 4x₁² + 4x₂²', fontsize=12)
    ax2.set_ylabel('f₂(x) = (x₁-5)² + (x₂-5)²', fontsize=12)
    ax2.set_title('Pareto Front by Method', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    
    # Plot 3: Worst-Case Score Bar Chart
    ax3 = plt.subplot(2, 3, 3)
    indices = list(range(1, len(solutions) + 1))
    scores = [mm['worst_case_score'] for mm in minimax_results]
    
    bars = ax3.bar(indices, scores, color=robustness_colors, edgecolor='black', linewidth=2)
    
    ax3.axhline(y=0.05, color='green', linestyle='--', alpha=0.6, 
                linewidth=2, label='Highly Robust')
    ax3.axhline(y=0.15, color='orange', linestyle='--', alpha=0.6, 
                linewidth=2, label='Moderately Robust')
    ax3.axhline(y=0.30, color='red', linestyle='--', alpha=0.6, 
                linewidth=2, label='Weakly Robust')
    
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
    
    bars = ax4.barh(indices, f1_rel, color=robustness_colors, edgecolor='black', linewidth=2)
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
    
    bars = ax5.barh(indices, f2_rel, color=robustness_colors, edgecolor='black', linewidth=2)
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
    
    # Add robustness legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Highly Robust'),
        Patch(facecolor='yellow', label='Moderately Robust'),
        Patch(facecolor='orange', label='Weakly Robust'),
        Patch(facecolor='red', label='Not Robust')
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
              ncol=4, fontsize=11, bbox_to_anchor=(0.5, 0.98))
    
    plt.suptitle('BNH Problem: Pareto Front with Minimax Stability Analysis (PySMT/Z3)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if not inf == timeout:
        timer = fig.canvas.new_timer(interval=timeout, callbacks=[(plt.close, [], {})])
        timer.start()
    
    plt.show()
    
    logging.info("\n7. Pareto Front Plot Generated")
    logging.info("-" * 60)
    logging.info("\n" + "=" * 60)
    logging.info(f"Found {len(solutions)} Pareto optimal solutions")
    for method, data in methods.items():
        logging.info(f"  - {method_styles.get(method, {}).get('label', method)}: {len(data['solutions'])} points")
    logging.info("=" * 60)
    logging.info("\n✓ Visualization complete")


if __name__ == "__main__":
    rootpath = "." if len(argv) < 2 else argv[1]
    timeout = inf if len(argv) < 3 else float(argv[2])
    print(solve_bnh_pysmt(rootpath, timeout))

