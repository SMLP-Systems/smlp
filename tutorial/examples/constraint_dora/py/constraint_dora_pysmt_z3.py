#!/usr/bin/python3.12
"""
Constrained Optimization using PySMT with Z3
Minimize: f(x) = (x1 - 2)^2 + (x2 - 1)^2
Subject to: 1 - x1^2 - x2^2 >= 0 (i.e., x1^2 + x2^2 <= 1)

This version uses Z3 instead of MathSAT.
Z3 is easier to install: pip install z3-solver pysmt
"""
from pysmt.shortcuts import Symbol, Real, GE, LE, And, Solver, get_env
from pysmt.typing import REAL
from pysmt.exceptions import NoSolverAvailableError

from hashlib import sha256

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


def objective_function(x1, x2):
    """Calculate the objective function value"""
    return (x1 - 2)**2 + (x2 - 1)**2


def check_solver_availability():
    """Check if Z3 is available"""
    env = get_env()
    available = list(env.factory.all_solvers())
    
    logging.info("="*60)
    logging.info("SOLVER AVAILABILITY CHECK")
    logging.info("="*60)
    logging.info(f"Available solvers: {available}")
    logging.info(f"Z3 available: {'z3' in available}")
    logging.info("="*60)
    
    if 'z3' not in available:
        logging.info("\n❌ Z3 not available in PySMT")
        logging.info("\nTo install Z3:")
        logging.info("  pip install z3-solver pysmt --break-system-packages")
        logging.info("\nAlternative: Use SciPy version:")
        logging.info("  python constrained_optimization_scipy.py")
        return False
    
    return True


def solve_constrained_optimization():
    """
    Solve the constrained optimization problem using PySMT with Z3.
    """
    
    # Define symbolic variables
    x1 = Symbol("x1", REAL)
    x2 = Symbol("x2", REAL)
    
    # Define the constraint: 1 - x1^2 - x2^2 >= 0
    constraint = GE(Real(1) - (x1 * x1) - (x2 * x2), Real(0))
    
    # Binary search bounds
    min_bound = 0.0
    max_bound = objective_function(0, 0)  # = 5
    
    epsilon = 0.0001
    best_x1 = None
    best_x2 = None
    best_value = float('inf')
    
    logging.info("\nSearching for optimal solution...")
    logging.info(f"Initial bounds: [{min_bound:.4f}, {max_bound:.4f}]")
    
    iteration = 0
    max_iterations = 100
    
    # Binary search on objective value
    while max_bound - min_bound > epsilon and iteration < max_iterations:
        iteration += 1
        mid = (min_bound + max_bound) / 2
        
        # Objective constraint: (x1 - 2)^2 + (x2 - 1)^2 <= mid
        obj_constraint = LE((x1 - Real(2)) * (x1 - Real(2)) + 
                           (x2 - Real(1)) * (x2 - Real(1)), 
                           Real(mid))
        
        # Combine constraints
        full_constraint = And(constraint, obj_constraint)
        
        # Check satisfiability with Z3
        with Solver(name="z3", logic="QF_NRA") as solver:
            solver.add_assertion(full_constraint)
            
            if solver.solve():
                # Found a feasible solution with objective <= mid
                model = solver.get_model()
                x1_val = float(model.get_value(x1).constant_value())
                x2_val = float(model.get_value(x2).constant_value())
                obj_val = objective_function(x1_val, x2_val)
                
                # Update best solution
                if obj_val < best_value:
                    best_value = obj_val
                    best_x1 = x1_val
                    best_x2 = x2_val
                
                # Search for smaller objective values
                max_bound = mid
                if iteration % 10 == 0:
                    logging.info(f"Iteration {iteration}: Found solution at f(x)={obj_val:.6f}")
            else:
                # No feasible solution with objective <= mid
                min_bound = mid
    
    logging.info(f"\nCompleted in {iteration} iterations")
    
    return best_x1, best_x2, best_value


def verify_solution(x1, x2):
    """Verify that the solution satisfies the constraint"""
    constraint_value = 1 - x1**2 - x2**2
    objective_value = objective_function(x1, x2)
    
    logging.info("\n" + "="*60)
    logging.info("SOLUTION VERIFICATION")
    logging.info("="*60)
    result = f"x1 = {x1:.6f}\nx2 = {x2:.6f}\n\nObjective function f(x) = {objective_value:.6f}"
    logging.info(result)
    logging.info(f"Constraint value (1 - x1² - x2²) = {constraint_value:.6f}")
    logging.info(f"Constraint satisfied: {constraint_value >= -1e-6}")
    logging.info("\nAnalytical solution (for comparison):")
    logging.info("  x1 ≈ 0.894427, x2 ≈ 0.447214, f(x) ≈ 1.527864")
    logging.info("  (Point on unit circle closest to (2,1))")
    logging.info("="*60)
    return result

def main(rootpath: str = ".", timeout: int = 30000):
    try:
        logging.info("="*60)
        logging.info("CONSTRAINED OPTIMIZATION WITH PYSMT AND Z3")
        logging.info("="*60)
        logging.info("Objective: minimize f(x) = (x1-2)² + (x2-1)²")
        logging.info("Subject to: x1² + x2² ≤ 1")
        logging.info("="*60)
        
        # Check if Z3 is available
        if not check_solver_availability():
            import sys
            sys.exit(1)
        
        # Solve the optimization problem
        x1_opt, x2_opt, f_opt = solve_constrained_optimization()
        
        # Verify and display results
        result = verify_solution(x1_opt, x2_opt)
        
    except NoSolverAvailableError as e:
        logging.error(f"\n❌ Solver Error: {e}")
        logging.error("\nTo install Z3:")
        logging.error("  pip install z3-solver pysmt --break-system-packages")
        return "Z3 failed"
        
    except Exception as e:
        logging.info(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        logging.error("\nFor a working alternative, try:")
        logging.error("  python constrained_optimization_scipy.py")
        return "Z3 failed"

    return sha256(str(result).encode()).hexdigest()

if __name__ == "__main__":
    print(main())
