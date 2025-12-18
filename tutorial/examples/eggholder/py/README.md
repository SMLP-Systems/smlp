# Eggholder Function Optimization

This project demonstrates global optimization techniques using SciPy on the Eggholder function, a complex mathematical function commonly used for testing optimization algorithms.

## Overview

The [Eggholder function]( https://www.sfu.ca/~ssurjano/egg.html ) is a challenging optimization problem with many local minima, making it an excellent benchmark for testing global optimization algorithms.<br>
![Eggholder function figure](eggholder_figure.png)<br><br>
![Eggholder function formula](eggholder_formula.png)<br><br>
Global minimum: _f(x*) = -959.6407, at x* = (512, 404.2319)_

This implementation:

- Visualizes the Eggholder function in 3D
- Generates a dataset of function values
- Compares two global optimization methods: SHGO and Dual Annealing
- Sorts and displays results

## Files

- **optimization_ex.py** - Main script that visualizes and optimizes the Eggholder function
- **sortdf.py** - Utility script for sorting dataframe files by column

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Running the Optimization

```bash
python3 optimization_ex.py
```

This will:
1. Generate a 3D plot of the Eggholder function (closes automatically after 5 seconds)
2. Create `dataset.txt` containing X1, X2, Y1 coordinates
3. Sort the dataset by the objective function value
4. Run two optimization algorithms and compare results

### Validation: creating dataset and optimization using SHGO
```bash
pytest
```
