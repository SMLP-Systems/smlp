## Mutltivarate constrained problem
[Binh and Korn (BNH) Multi-Objective Optimization Problem](https://pymoo.org/problems/multi/bnh.html)
### Problem Definition:
```
Minimize:
    f₁(x) = 4x₁² + 4x₂²
    f₂(x) = (x₁ - 5)² + (x₂ - 5)²
Subject to:
    C₁(x) = (x₁ - 5)² + x₂² ≤ 25
    C₂(x) = (x₁ - 8)² + (x₂ + 3)² ≥ 7.7
    0 ≤ x₁ ≤ 5
    0 ≤ x₂ ≤ 3
```
Expected and NSGA-II approximation results:
![ResultsImage](media/BNH.png)
