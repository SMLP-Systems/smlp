# Binh and Korn (BNH) Multi-Objective Optimization Problem

[Binh and Korn (BNH) Multi-Objective Optimization Problem](https://pymoo.org/problems/multi/bnh.html)

## Multivariate Constrained Problem

### Problem Definition

**Minimize:**

```math
    f₁(x) = 4x₁² + 4x₂² 
    f₂(x) = (x₁ - 5)² + (x₂ - 5)²
```

**Subject to:**

```math
    C₁(x) = (x₁ - 5)² + x₂² ≤ 25 
    C₂(x) = (x₁ - 8)² + (x₂ + 3)² ≥ 7.7
    0 ≤ x₁ ≤ 5 
    0 ≤ x₂ ≤ 3
```

### Expected and NSGA-II approximation results:
![ResultsImage](media/BNH.png)
#### Z3 versions used:
- Z3 version: 4.8.12,  64 bit
- Python package version: z3-solver==4.8.12<br>
### Expected Z3 results:
![ResultsImage](media/BNH_Z3.png)
### Expected Z3 gradient-stable results (3 flavors):
![ResultsImage](media/BNH_Z3.stable_gradient_flavor1.png)
![ResultsImage](media/BNH_Z3.stable_gradient_flavor2.png)
![ResultsImage](media/BNH_Z3.stable_gradient_flavor3.png)
### Expected Z3 minimax-stable results (2 flavors):
![ResultsImage](media/BNH_Z3.stable_minimax_flavor1.png)<br><br>
![ResultsImage](media/BNH_Z3.stable_minimax_flavor2.png)
