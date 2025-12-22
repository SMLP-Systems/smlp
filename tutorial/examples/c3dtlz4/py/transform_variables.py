#!/usr/bin/python3.12

from mpmath import pi, cos, acos, power, mpf, mp, sqrt, fsum as sum
from pandas import read_csv, isna
from numpy import floating
from sys import argv
from os.path import realpath, dirname

def dtlz4_objectives_t(y, n_objectives=2, alpha=mpf(100.0)):
    n_vars = len(y)
    k = n_vars - n_objectives + 1  # Number of variables in g function
    
    # Calculate g function using the last k variables
    y_M = y[n_objectives - 1:]  # Last k variables
    g = sum(power(power( acos(yi) * mpf(2.0) / pi, mpf(1.0)/alpha) - mpf(1.0)/mpf(2.0), mpf(2)) for yi in y_M)
    
    # Calculate objectives
    objectives = []
    
    for i in range(n_objectives):
        f = 1.0 + g
        
        # Multiply by cosine terms
        for j in range(n_objectives - i - 1):
            f *= y[j]
        
        # Multiply by sine term (if not the first objective)
        if i > 0:
            f *= sqrt(mpf(1.0) - power(y[n_objectives - i - 1],mpf(2)))
        
        objectives.append(f)
    
    return objectives
    
def transform_objectives(csv: str = "results.csv", alpha: mpf = mpf(100)):
    df = read_csv(csv,sep=',')
    dim = len(df.columns[df.columns.str.startswith('X')])
    traformed_variables_f=realpath(dirname(csv)) + "/transformed_variables.csv"
    with open(traformed_variables_f, "w") as tv:
        tv.write("N,")
        [tv.write(f"X{i},") for i in range(0, dim)]
        [tv.write(f"Y{i},") for i in range(0, dim)]
        tv.write(f"F1,F2\n")
        for i in range(0,df.shape[0]):
            tv.write(f"{df['N'][i]:3d},")
            x=[]
            y=[]
            [x.append(df['X'+str(j)][i]) for j in range(0, dim)]
            [y.append(cos(power(mpf(x[j]),alpha) * pi / mpf(2.0))) for j in range(0, dim)]
            [tv.write(f"{x[j]},") for j in range(0, dim)]
            [tv.write(f"{float(y[j]):.16g},") for j in range(0, dim-1)]
            tv.write(f"{float(y[dim-1]):.16g},")
            objectives_t=dtlz4_objectives_t(y)
            tv.write(f"{float(objectives_t[0]):.16g},{float(objectives_t[1]):.16g}\n")
    return 0
if __name__ == "__main__":
    mp.dps=500
    results = "results.csv" if len(argv) < 2 else argv[1]
    print(transform_objectives(results))
