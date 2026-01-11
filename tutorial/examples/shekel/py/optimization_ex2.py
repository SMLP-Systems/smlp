#!/usr/bin/python3.14
import numpy as np
from scipy.optimize import shgo
from hashlib import sha256
from csv import writer

def shekel_function(x):
    """
    Shekel function implementation.
    
    f(x) = -sum_{i=1}^{m} (sum_{j=1}^{4}(x_j - C_{ij})^2 + beta_i)^{-1}
    
    Parameters:
    -----------
    x : array-like, shape (4,)
        Input vector with 4 dimensions
    
    Returns:
    --------
    float
        Function value at x
    """
    # Define parameters
    m = 10
    
    # Beta vector (reading from image: 1,2,2,4,4,6,3,7,5,5 with superscript T)
    beta = np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5]) / 10.0
    
    # C matrix (4 x m) - reading directly from the image
    # Row 1 (j=1): 4.0  1.0  8.0  6.0  3.0  2.0  5.0  8.0  6.0  7.0
    # Row 2 (j=2): 4.0  1.0  8.0  6.0  7.0  9.0  3.0  1.0  2.0  3.6
    # Row 3 (j=3): 4.0  1.0  8.0  6.0  3.0  2.0  5.0  8.0  6.0  7.0
    # Row 4 (j=4): 4.0  1.0  8.0  6.0  7.0  9.0  3.0  1.0  2.0  3.6
    C = np.array([
        [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
        [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
        [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
        [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]
    ])
    
    # Compute the function value
    x = np.array(x)
    result = 0.0
    
    for i in range(m):
        inner_sum = 0.0
        for j in range(4):
            inner_sum += (x[j] - C[j, i])**2
        result += 1.0 / (inner_sum + beta[i])
    
    return -result

def create_meshgrid_csv(n_points=10, filename='shekel_meshgrid.csv'):
    """
    Create a CSV file with Shekel function values on a meshgrid.
    
    Parameters:
    -----------
    n_points : int
        Number of points along each dimension (default: 10)
    filename : str
        Output CSV filename (default: 'shekel_meshgrid.csv')
    """
    # Create 1D arrays for each dimension
    x1 = np.linspace(0, 10, n_points)
    x2 = np.linspace(0, 10, n_points)
    x3 = np.linspace(0, 10, n_points)
    x4 = np.linspace(0, 10, n_points)
    
    # Create meshgrid
    X1, X2, X3, X4 = np.meshgrid(x1, x2, x3, x4, indexing='ij')
    
    # Stack all coordinates into a single array using column_stack
    # Shape: (n_points^4, 4) where each row is [x1, x2, x3, x4]
    points = np.column_stack([X1.flatten(), X2.flatten(), X3.flatten(), X4.flatten()])
    
    print(f"Creating CSV with {len(points)} points...")
    print(f"Grid size: {n_points} points per dimension")
    print(f"Total points: {n_points**4}")
    
    # Calculate all function values
    f_values = np.array([shekel_function(point) for point in points])
    
    # Open CSV file for writing
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = writer(csvfile)
        csv_writer.writerow(['X1', 'X2', 'X3', 'X4', 'Y'])
        #for i, point in enumerate(points):
        #    writer.writerow([point[0], point[1], point[2], point[3], f_values[i]])
        [csv_writer.writerow([point[0], point[1], point[2], point[3], f_values[i]]) for i, point in enumerate(points)]
    
    print(f"\nCSV file '{filename}' created successfully!")
    print(f"Total rows: {len(points) + 1} (including header)")
    
    # Find and display the minimum value
    min_idx = np.argmin(f_values)
    min_value = f_values[min_idx]
    min_x = points[min_idx]
    
    print(f"\nMinimum value found in grid:")
    print(f"  X = {min_x}")
    print(f"  Y = {min_value:.10f}")

def optimize_shekel():
    """
    Optimize the Shekel function using SHGO algorithm.
    """
    # Define bounds for each dimension (typically [0, 10] for Shekel function)
    bounds = [(0, 10)] * 4
    
    print("Optimizing Shekel function using SHGO algorithm...")
    print("=" * 60)
    
    result = shgo(shekel_function, bounds, n=200, iters=5)

    # Display results
    print(f"\nOptimization Results:")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"\nOptimal solution (x*):")
    print(f"  x = {result.x}")
    print(f"\nOptimal function value (f(x*)):")
    print(f"  f(x*) = {result.fun:.10f}")
    print(f"\nNumber of function evaluations: {result.nfev}")
    print(f"Number of iterations: {result.nit}")
    
    # The known global minimum for Shekel function is approximately:
    # x* ≈ [4, 4, 4, 4] and f(x*) ≈ -10.5363
    print(f"\n" + "=" * 60)
    print("Note: The known global minimum is approximately:")
    print("  x* ≈ [4, 4, 4, 4]")
    print("  f(x*) ≈ -10.5363")
    
    return result

def main(n: int = 512, rootpath: str = ".") -> int:
    # Run optimization
    result = optimize_shekel()
    
    # Test the function at a few points
    print("\n" + "=" * 60)
    print("Function evaluations at test points:")
    test_points = [
        [4.0, 4.0, 4.0, 4.0],
        [1.2, 2.0, 3.2, 4.0],
        [8.0, 7.2, 6.0, 4.8]
    ]
    
    for point in test_points:
        value = shekel_function(point)
        print(f"  f({point}) = {value:.6f}")

    create_meshgrid_csv(n_points=26, filename=rootpath + '/shekel_meshgrid_26.csv')

    return sha256(str(result).encode()).hexdigest()

if __name__ == "__main__":
   print(main())
