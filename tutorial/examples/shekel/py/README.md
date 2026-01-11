# Shekel Function Optimization

## Overview

This repository contains an implementation of the Shekel function and its optimization using the SHGO (Simplicial Homology Global Optimization) algorithm from scipy.

## Shekel Function Definition

The Shekel function is defined as:

```
f(x) = -∑(i=1 to m) [∑(j=1 to 4)(xj - Cij)² + βi]⁻¹
```

where:
- **m = 10** (number of terms in the outer sum)
- **x** is a 4-dimensional input vector: x = [x₁, x₂, x₃, x₄]
- **β** is a parameter vector: β = (1/10)(1, 2, 2, 4, 4, 6, 3, 7, 5, 5)ᵀ
- **C** is a 4×10 matrix:

```
C = [ 4.0  1.0  8.0  6.0  3.0  2.0  5.0  8.0  6.0  7.0 ]
    [ 4.0  1.0  8.0  6.0  7.0  9.0  3.0  1.0  2.0  3.6 ]
    [ 4.0  1.0  8.0  6.0  3.0  2.0  5.0  8.0  6.0  7.0 ]
    [ 4.0  1.0  8.0  6.0  7.0  9.0  3.0  1.0  2.0  3.6 ]
```

## Global Minimum

The known global minimum of the Shekel function is approximately:
- **x* ≈ [4, 4, 4, 4]**
- **f(x*) ≈ -10.5363**

## Requirements

```
numpy
scipy
```

## Installation

```bash
pip install numpy scipy
```

## Usage

Run the optimization script:

```python
python shekel_optimization.py
```

The script will:
1. Define the Shekel function with the specified parameters
2. Optimize it using the SHGO algorithm
3. Display the optimal solution and function value
4. Test the function at several points for validation

## SHGO Parameters

The optimization uses the following SHGO parameters for robust convergence:
- **n = 250**: Number of sampling points
- **iters = 10**: Number of iterations
- **sampling_method = 'sobol'**: Sobol sequence for better space coverage
- **ftol = 1e-10**: Function tolerance for convergence
- **minimize_every_iter = True**: Perform local minimization at each iteration

## Example Output

```
Optimizing Shekel function using SHGO algorithm...
============================================================

Optimization Results:
Success: True
Message: Optimization terminated successfully.

Optimal solution (x*):
  x = [4.00000000 4.00000000 4.00000000 4.00000000]

Optimal function value (f(x*)):
  f(x*) = -10.5362845316

Number of function evaluations: 2547
Number of iterations: 10
```

## Function Characteristics

- **Domain**: x ∈ [0, 10]⁴
- **Type**: Multimodal (multiple local minima)
- **Difficulty**: Moderate - requires global optimization methods
- **Applications**: Benchmark testing for optimization algorithms

## Notes

The Shekel function is commonly used as a benchmark for testing global optimization algorithms due to its multiple local minima. The SHGO algorithm is particularly effective for this function as it uses topological methods to identify and explore multiple basins of attraction.

## Reference
[Shekel function](https://www.sfu.ca/~ssurjano/shekel.html)
