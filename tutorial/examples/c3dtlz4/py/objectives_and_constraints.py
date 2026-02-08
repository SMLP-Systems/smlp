#!/usr/bin/python3.12
"""
C3-DTLZ4 Objective Functions
Based on the DTLZ test suite with DTLZ4 objectives and C3 constraints.

Reference:
- Deb et al. (2002) - Scalable Test Problems for Evolutionary Multiobjective Optimization
- Jain & Deb (2014) - An Evolutionary Many-Objective Optimization Algorithm (for constraints)
"""

from math import pi, sin, cos
from pandas import read_csv, isna
from numpy import floating
from sys import argv
from os.path import realpath, dirname

def compare_dataframes(df1, df2, float_threshold=0.01, report_file=None):
    """
    Compare two DataFrames and report mismatches.
    
    Parameters:
    -----------
    df1 : pandas.DataFrame
        First DataFrame
    df2 : pandas.DataFrame
        Second DataFrame
    float_threshold : float
        Relative threshold for floating point comparisons (default: 0.01 = 1%)
    report_file : str, optional
        Path to save the report. If None, returns the report as string
    
    Returns:
    --------
    str : Comparison report
    """
    
    report = []
    report.append("=" * 80)
    report.append("DATAFRAME COMPARISON REPORT")
    report.append("=" * 80)
    
    # Check shape
    if df1.shape != df2.shape:
        report.append(f"\n⚠ SHAPE MISMATCH:")
        report.append(f"  DataFrame 1: {df1.shape}")
        report.append(f"  DataFrame 2: {df2.shape}")
        report.append("\nCannot proceed with detailed comparison due to shape mismatch.")
        result = "\n".join(report)
        if report_file:
            with open(report_file, 'w') as f:
                f.write(result)
        return result
    
    # Check column names
    if not df1.columns.equals(df2.columns):
        report.append(f"\n⚠ COLUMN MISMATCH:")
        report.append(f"  Only in df1: {set(df1.columns) - set(df2.columns)}")
        report.append(f"  Only in df2: {set(df2.columns) - set(df1.columns)}")
    
    # Check index
    if not df1.index.equals(df2.index):
        report.append(f"\n⚠ INDEX MISMATCH:")
        report.append(f"  Indices are different")
    
    # Compare each column
    total_mismatches = 0
    
    for col in df1.columns:
        if col not in df2.columns:
            continue
            
        col_mismatches = []
        
        for idx in df1.index:
            if idx not in df2.index:
                continue
                
            val1 = df1.loc[idx, col]
            val2 = df2.loc[idx, col]
            
            # Handle NaN values
            if isna(val1) and isna(val2):
                continue
            elif isna(val1) or isna(val2):
                col_mismatches.append((idx, val1, val2, "NaN mismatch"))
                continue
            
            # Determine data type and compare
            is_float1 = isinstance(val1, (float, floating))
            is_float2 = isinstance(val2, (float, floating))
            
            if is_float1 or is_float2:
                # Floating point comparison with relative threshold
                if val1 == 0 and val2 == 0:
                    continue
                elif val1 == 0 or val2 == 0:
                    # If one is zero, use absolute difference
                    if abs(val1 - val2) > float_threshold:
                        rel_diff = abs(val1 - val2)
                        col_mismatches.append((idx, val1, val2, f"abs_diff={rel_diff:.3e}"))
                else:
                    # Calculate relative difference
                    rel_diff = abs(val1 - val2) / abs(val1)
                    if rel_diff > float_threshold:
                        col_mismatches.append((idx, val1, val2, f"rel_diff={rel_diff:.3e}"))
            else:
                # Integer or other types - exact match required
                if val1 != val2:
                    col_mismatches.append((idx, val1, val2, "exact mismatch"))
        
        # Report column mismatches
        if col_mismatches:
            total_mismatches += len(col_mismatches)
            report.append(f"\n{'─' * 80}")
            report.append(f"Column: '{col}' - {len(col_mismatches)} mismatch(es)")
            report.append(f"{'─' * 80}")
            
            for idx, v1, v2, reason in col_mismatches[:20]:  # Limit to first 20
                report.append(f"  Line {idx+2}: {v1} != {v2} ({reason})")
            
            if len(col_mismatches) > 20:
                report.append(f"  ... and {len(col_mismatches) - 20} more mismatches")
    
    # Summary
    report.append(f"\n{'=' * 80}")
    report.append(f"SUMMARY:")
    report.append(f"  Total mismatches found: {total_mismatches}")
    report.append(f"  Float threshold: {float_threshold * 100}%")
    if total_mismatches == 0:
        report.append(f"  ✓ DataFrames match!")
    report.append(f"{'=' * 80}")
    
    result = "\n".join(report)
    
    if report_file:
        with open(report_file, 'w') as f:
            f.write(result)
    
    return result

def dtlz4_objectives(x, n_objectives=2, alpha=100.0):
    """
    DTLZ4 objective functions.
    
    DTLZ4 is similar to DTLZ2 but uses a parameter alpha to control
    the density of solutions. Higher alpha creates a more biased density.
    
    Args:
        x: Decision variables [x0, x1, ..., x_{n-1}], all in [0, 1]
        n_objectives: Number of objectives (m)
        alpha: Density parameter (default: 100.0)
    
    Returns:
        List of objective values [f1, f2, ..., fm]
    
    Formula:
        g(x_M) = sum((x_i - 0.5)^2 for i in x_M)
        
        For m objectives:
        f_1 = (1 + g) * cos(x_1^alpha * π/2) * cos(x_2^alpha * π/2) * ... * cos(x_{m-1}^alpha * π/2)
        f_2 = (1 + g) * cos(x_1^alpha * π/2) * cos(x_2^alpha * π/2) * ... * sin(x_{m-1}^alpha * π/2)
        ...
        f_{m-1} = (1 + g) * cos(x_1^alpha * π/2) * sin(x_2^alpha * π/2)
        f_m = (1 + g) * sin(x_1^alpha * π/2)
        
    where x_M are the last k = n - m + 1 variables used for the g function.
    """
    n_vars = len(x)
    k = n_vars - n_objectives + 1  # Number of variables in g function
    
    # Calculate g function using the last k variables
    x_M = x[n_objectives - 1:]  # Last k variables
    g = sum((xi - 0.5) ** 2 for xi in x_M)
    
    # Calculate objectives
    objectives = []
    
    for i in range(n_objectives):
        f = 1.0 + g
        
        # Multiply by cosine terms
        for j in range(n_objectives - i - 1):
            f *= cos(x[j] ** alpha * pi / 2.0)
        
        # Multiply by sine term (if not the first objective)
        if i > 0:
            f *= sin(x[n_objectives - i - 1] ** alpha * pi / 2.0)
        
        objectives.append(f)
    
    return objectives

def c3dtlz4_constraints(objectives):
    """
    C3-DTLZ4 constraint functions.
    
    For each objective f_i:
        c_i = -(sum of all f_j²) + 0.75 * f_i + 1.0
    
    Args:
        objectives: List of objective values [f1, f2, ..., fm]
    
    Returns:
        List of constraint values [c1, c2, ..., cm]
        Feasible if ALL constraints ≤ 0
    """
    sum_squares = sum(f**2 for f in objectives)
    constraints = [-sum_squares + f * 0.75 + 1.0 for f in objectives]
    return constraints


def evaluate_c3dtlz4(x, n_objectives=2, alpha=100.0):
    """
    Complete evaluation of C3-DTLZ4 problem.
    
    Args:
        x: Decision variables [x0, x1, ..., x_{n-1}], all in [0, 1]
        n_objectives: Number of objectives
        alpha: Density parameter (default: 100.0)
    
    Returns:
        Dictionary with:
            - 'objectives': List of objective values
            - 'constraints': List of constraint values
            - 'feasible': Boolean indicating if solution is feasible
    """
    objectives = dtlz4_objectives(x, n_objectives, alpha)
    constraints = c3dtlz4_constraints(objectives)
    feasible = all(c <= 0 for c in constraints)
    
    return {
        'objectives': objectives,
        'constraints': constraints,
        'feasible': feasible
    }

def objectives_and_constraints(csv: str = "results.csv", rel_thr: float = 0.0001) -> str:
    df = read_csv(csv,sep=',')
    dim = len(df.columns[df.columns.str.startswith('X')])
    obj_const_f=realpath(dirname(csv)) + "/obj_const.csv"
    with open(obj_const_f, "w") as o:
        o.write("N,")
        [o.write(f"X{i},") for i in range(0, dim)]
        o.write("F1,F2,C1,C2\n")
        for i in range(0,df.shape[0]):
            x=[]
            [x.append(df['X'+str(j)][i]) for j in range(0, dim)]
            result = evaluate_c3dtlz4(x)
            o.write(f"{df['N'][i]:3d},")
            [o.write(f"{x[j]},") for j in range(0, dim)]
            o.write(f"{result['objectives'][0]},{result['objectives'][1]},{result['constraints'][0]},{result['constraints'][1]}\n")
    df1 = read_csv(obj_const_f,sep=',')
    return compare_dataframes(df, df1, float_threshold=rel_thr)

if __name__ == "__main__":
    results = "results.csv" if len(argv) < 2 else argv[1]
    print(objectives_and_constraints(results))
