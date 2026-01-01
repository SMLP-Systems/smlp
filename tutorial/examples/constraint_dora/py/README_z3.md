# Constrained Optimization Problem

## Problem Statement

**Minimize:**
```
f(x₁, x₂) = (x₁ - 2)² + (x₂ - 1)²
```

**Subject to:**
```
x₁² + x₂² - 1 ≤ 0  (inside unit circle)
```

## Solution Approaches

This repository demonstrates three different approaches to solving this constrained optimization problem.

### 1. Z3 Optimizer

Z3 is a theorem prover and constraint solver from Microsoft Research. While it can handle optimization problems, it's primarily designed for constraint satisfaction rather than continuous nonlinear optimization.

**Pros:**
- Can handle complex logical constraints
- Good for mixed integer/real problems
- Formal verification capabilities

**Cons:**
- May provide approximate solutions for nonlinear problems
- Not specifically optimized for continuous optimization

### 2. Scipy SLSQP (Recommended)

The `scipy.optimize.minimize` function with the SLSQP (Sequential Least Squares Programming) method is the recommended approach for this type of problem.

**Pros:**
- Designed specifically for constrained nonlinear optimization
- Fast and accurate
- Well-tested and widely used

**Why it's better for this problem:**
- Handles smooth nonlinear objectives efficiently
- Native support for inequality constraints
- Provides accurate numerical solutions

### 3. Analytical Solution

For this specific problem, we can derive the optimal solution analytically.

## Mathematical Insight

The problem asks us to find the point inside or on the unit circle that is closest to the point (2, 1).

**Key observations:**
1. The objective function `(x₁ - 2)² + (x₂ - 1)²` represents the squared Euclidean distance from (x₁, x₂) to (2, 1)
2. The point (2, 1) lies outside the unit circle since 2² + 1² = 5 > 1
3. Therefore, the minimum distance occurs at a point on the boundary of the circle (not inside)
4. The optimal point lies on the line segment from the origin to (2, 1)

**Analytical solution:**

The optimal point is found by normalizing the vector (2, 1):

```
direction = (2, 1) / ||(2, 1)|| = (2, 1) / √5

Optimal point: (x₁*, x₂*) = (2/√5, 1/√5) ≈ (0.894427, 0.447214)

Minimum value: f(x₁*, x₂*) = (√5 - √5)² = (√5 - 1)² ≈ 1.236068
```

Wait, let me recalculate:
```
f(2/√5, 1/√5) = (2/√5 - 2)² + (1/√5 - 1)²
              = ((2 - 2√5)/√5)² + ((1 - √5)/√5)²
              = (2 - 2√5)²/5 + (1 - √5)²/5
              = [(2 - 2√5)² + (1 - √5)²]/5
              = (√5 - 1)² ≈ 1.236068
```

## Installation

```bash
pip install z3-solver scipy numpy
```

## Usage

Run the Python script to see all three solutions:

```bash
python constraint_dora_z3.py
```

## Expected Output

The script will display:
1. Z3 solution with coordinates and objective value
2. Scipy solution with coordinates and objective value
3. Analytical solution with exact mathematical derivation
4. Verification that all solutions satisfy the constraint

## Files

- `constraint_dora_z3.py` - Main Python script with all three solution methods
- `README.md` - This file

## Conclusion

For this type of smooth, continuous constrained optimization problem, **scipy's SLSQP method is recommended** for practical applications. Z3 is shown for educational purposes to demonstrate its optimization capabilities, while the analytical solution provides mathematical insight into the problem structure.

**Reference**

- Claude Sonnet 4.5 free version
