# Non-linear Constraints: Cattle Feed Problem (HS73)

## Problem Description

This is a classic non-linear optimization problem from Hock and Schittkowski's test problem collection (problem 73), also known as the cattle-feed problem. It demonstrates constrained optimization with both linear and non-linear constraints.

## Mathematical Formulation

### Objective Function
Minimize:<br> `f(x) = 24.55x₁ + 26.75x₂ + 39x₃ + 40.50x₄`

### Constraints

**Linear Constraint:**<br>
`2.3x₁ + 5.6x₂ + 11.1x₃ + 1.3x₄ - 5 ≥ 0`

**Non-linear Constraint:**<br>
`12x₁ + 11.9x₂ + 41.8x₃ + 52.1x₄ - 1.645√(0.28x₁² + 0.19x₂² + 20.5x₃² + 0.62x₄²) - 21 ≥ 0`

**Equality Constraint:**<br>
`x₁ + x₂ + x₃ + x₄ - 1 = 0`

**Bounds:**<br>
`0 ≤ xᵢ ≤ 1` for all i ∈ {1, 2, 3, 4}

## Problem Context

This problem represents a cattle feed mixture optimization where:
- **Decision variables (x₁, x₂, x₃, x₄)**: proportions of four different feed ingredients
- **Objective**: Minimize the total cost of the feed mixture
- **Constraints**: Ensure nutritional requirements are met while maintaining mixture proportions

The equality constraint ensures that the proportions sum to 1 (100% of the mixture).

## Optimal Solution

The approximate optimal solution is:

```
x₁ ≈ 0.6355216
x₂ ≈ -0.12 × 10⁻¹¹ (essentially 0)
x₃ ≈ 0.3127019
x₄ ≈ 0.05177655

f(x*) ≈ 29.894378
```

## Key Features

- **Problem Type**: Non-linear programming (NLP)
- **Difficulty**: The non-linear constraint involving a square root of sum of squares makes this problem non-convex
- **Dimensions**: 4 variables, 4 constraints (3 inequality, 1 equality)
- **Applications**: Feed formulation, mixture problems, portfolio optimization

## Implementation Notes

To solve this problem, you would typically use:
- Non-linear optimization solvers (e.g., IPOPT, SNOPT, fmincon)
- Sequential Quadratic Programming (SQP) methods
- Interior point methods

The problem requires careful handling of the non-linear constraint, especially ensuring the expression under the square root remains non-negative during optimization.

## References

[1] Hock, W. and Schittkowski, K., "Test Examples for Nonlinear Programming Codes", Lecture Notes in Economics and Mathematical Systems, Vol. 187, Springer-Verlag, 1981.