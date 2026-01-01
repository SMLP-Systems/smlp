#!/usr/bin/python3.12
from z3 import *
import numpy as np
from scipy.optimize import minimize
from math import nan
from sys import argv
from hashlib import sha256

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

def main(rootpath: str = ".", timeout: int = 30000):
    logging.info("=" * 60)
    logging.info("METHOD 1: Z3 Optimizer (for comparison)")
    logging.info("=" * 60)
    results = {}
    
    # Create Z3 optimizer
    opt = Optimize()
    
    # Define real variables
    x1 = Real('x1')
    x2 = Real('x2')
    
    # Define the objective function: (x1-2)^2 + (x2-1)^2
    objective = (x1 - 2)**2 + (x2 - 1)**2
    
    # Add constraint: x1^2 + x2^2 <= 1 (inside unit circle)
    opt.add(x1**2 + x2**2 <= 1)
    
    # Minimize the objective
    opt.set("timeout",timeout)
    opt.minimize(objective)
    
    # Solve
    if opt.check() == sat:
        model = opt.model()
        x1_val = model[x1]
        x2_val = model[x2]
        
        # Convert to float for display
        x1_float = float(x1_val.as_decimal(10).rstrip('?'))
        x2_float = float(x2_val.as_decimal(10).rstrip('?'))
        obj_val = (x1_float - 2)**2 + (x2_float - 1)**2
        
        logging.info(f"Z3 Solution:")
        logging.info(f"  x1 = {x1_float:.6f}")
        logging.info(f"  x2 = {x2_float:.6f}")
        logging.info(f"  f(x1, x2) = {obj_val:.6f}")
        logging.info(f"  Constraint check: x1^2 + x2^2 = {x1_float**2 + x2_float**2:.6f}")
        results['z3']=(round(x1_float,4),round(x2_float,4))
    else:
        logging.info(f"No solution found by Z3 after {timeout//1000} second(s)")
        results['z3']=(nan,nan)
     
    logging.info("\n" + "=" * 60)
    logging.info("METHOD 2: Scipy (Recommended for this problem)")
    logging.info("=" * 60)
    
    # Define objective function
    def objective_func(x):
        return (x[0] - 2)**2 + (x[1] - 1)**2
    
    # Define constraint: x1^2 + x2^2 - 1 <= 0
    def constraint_func(x):
        return 1 - (x[0]**2 + x[1]**2)  # Must be >= 0
    
    constraint = {'type': 'ineq', 'fun': constraint_func}
    
    # Initial guess
    x0 = [0.5, 0.5]
    
    # Solve using SLSQP (Sequential Least Squares Programming)
    result = minimize(objective_func, x0, method='SLSQP', constraints=constraint)
    
    logging.info(f"Scipy Solution:")
    logging.info(f"  x1 = {result.x[0]:.6f}")
    logging.info(f"  x2 = {result.x[1]:.6f}")
    logging.info(f"  f(x1, x2) = {result.fun:.6f}")
    logging.info(f"  Constraint check: x1^2 + x2^2 = {result.x[0]**2 + result.x[1]**2:.6f}")
    logging.info(f"  Success: {result.success}")
    results['slsqp']=(float(round(result.x[0],4)),float(round(result.x[1],4)))
    
    logging.info("\n" + "=" * 60)
    logging.info("ANALYTICAL SOLUTION")
    logging.info("=" * 60)
    
    # The optimal point lies on the boundary of the circle
    # in the direction from origin to (2,1)
    target = np.array([2, 1])
    direction = target / np.linalg.norm(target)
    optimal = direction  # Scale to unit circle
    
    logging.info(f"Analytical Solution:")
    logging.info(f"  x1 = {optimal[0]:.6f}")
    logging.info(f"  x2 = {optimal[1]:.6f}")
    logging.info(f"  f(x1, x2) = {objective_func(optimal):.6f}")
    logging.info(f"  Constraint check: x1^2 + x2^2 = {optimal[0]**2 + optimal[1]**2:.6f}")
    
    logging.info("\n" + "=" * 60)
    logging.info("EXPLANATION")
    logging.info("=" * 60)
    logging.info("The minimum occurs at the point on the unit circle")
    logging.info("closest to (2, 1). This is found by moving from the")
    logging.info("origin toward (2, 1) until hitting the circle boundary.")
    logging.info("=" * 60)
    results['linalg']=(float(round(optimal[0],4)), float(round(optimal[1],4)))
    logging.info(results)
    return sha256(str(results).encode()).hexdigest()

if __name__ == "__main__":
    match len(argv):
        case 1:
            print(main())
        case 2:
            print(main(argv[1]))
        case _:
            print(main(argv[1],int(argv[2])))
