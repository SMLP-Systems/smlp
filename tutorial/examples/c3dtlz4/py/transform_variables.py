#!/usr/bin/python3.12

from mpmath import pi, cos, acos, power, mpf, mp, sqrt, fsum as sum
from pandas import read_csv, isna
from numpy import floating
from sys import argv
from os.path import realpath, dirname
from objectives_and_constraints import compare_dataframes
from study_results import study_results, add_study_arguments
from hashlib import sha256

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
    with open(tcsv, "w") as tv:
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
    args=add_study_arguments().parse_args()
    if study_results(args):
        results_csv = args.path + "/" + args.results
        mp.dps=500
        tcsv = transform_objectives(results_csv, args.path + "/transformed_variables.csv")
        df1=read_csv(results_csv,sep=',')
        df2=read_csv(tcsv,sep=',').drop(['Y0','Y1','Y2'],axis=1)
        comparison_results=compare_dataframes(df1,df2,0.001)
        print(comparison_results)
        return sha256(comparison_results.encode()).hexdigest()
    else:
        print("\nERROR: results analysis failed\n")
        return -1   
if __name__ == "__main__":
    print(main())
