#!/usr/bin/python3.12

from mpmath import pi, cos, acos, power, mpf, mp, sqrt, fsum as sum
from pandas import read_csv, isna
from numpy import floating
from sys import argv
from os.path import realpath, dirname
from objectives_and_constraints import compare_dataframes

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

def c3dtlz4_constraints(objectives):
    return [-objectives[0]**2 - objectives[1]**2 + objectives[0]*0.75 + 1.0,
            -objectives[0]**2 - objectives[1]**2 + objectives[1]*0.75 + 1.0]
    
def transform_objectives(csv: str = "results.csv", tcsv: str = "transformed_variables.csv", alpha: mpf = mpf(100)) -> bool:
    df = read_csv(csv,sep=',')
    dim = len(df.columns[df.columns.str.startswith('X')])
    traformed_variables_f=realpath(dirname(csv)) + "/" + tcsv
    with open(traformed_variables_f, "w") as tv:
        tv.write("N,")
        [tv.write(f"X{i},") for i in range(0, dim)]
        [tv.write(f"Y{i},") for i in range(0, dim)]
        tv.write(f"F1,F2,C1,C2\n")
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
            constraints_t=c3dtlz4_constraints(objectives_t)
            tv.write(f"{float(objectives_t[0]):.16g},{float(objectives_t[1]):.16g},")
            tv.write(f"{float(constraints_t[0]):.16g},{float(constraints_t[1]):.16g}\n")
    return tcsv

def main():
    mp.dps=500
    results = "results.csv" if len(argv) < 2 else argv[1]
    tcsv = transform_objectives(results)
    df1=read_csv(results,sep=',')
    df2=read_csv(tcsv,sep=',').drop(['Y0','Y1','Y2'],axis=1)
    print(compare_dataframes(df1,df2,0.001))
    return 0

if __name__ == "__main__":
   exit(main())
