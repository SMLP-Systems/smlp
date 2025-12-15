#!/usr/bin/python3.12
"""
https://docs.scipy.org/doc/scipy/tutorial/optimize.html#global-optimization
https://www.sfu.ca/~ssurjano/egg.html
"""
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from numpy import sin, sqrt, arange, meshgrid, stack, abs
from os import popen
from sys import argv
from os.path import realpath, dirname
from rich import print as rprint

eggholder = lambda x, k = 47: (-(x[1] + k) * sin(sqrt(abs(x[0]/2 + (x[1]  + k)))) 
                                     -x[0] * sin(sqrt(abs(x[0]   - (x[1]  + k)))))

def main(n = 512):

    r = range(-n, n+1)
    x = arange(r.start, r.stop)
    xgrid, ygrid = meshgrid(x, x)
    xy = stack([xgrid, ygrid])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    angle=45
    ax.view_init(angle, -angle)
    eggholder_xy=eggholder(xy)
    with open("dataset.txt","w") as ds:
        ds.write("X1 X2 Y1\n")
        [[ds.write(f"{xy[0][i][j]} {xy[1][i][j]} {eggholder_xy[i][j]}\n") for j in r] for i in r]
    ax.plot_surface(xgrid, ygrid, eggholder_xy, cmap='terrain')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('eggholder(x, y)')
    # set window title only if a canvas manager is available (some backends may not provide one)
    mgr = getattr(fig.canvas, "manager", None)
    if mgr is not None and hasattr(mgr, "set_window_title"):
        try:
            mgr.set_window_title('Eggholder plot')
        except Exception:
            pass
    timer = fig.canvas.new_timer(interval=5000, callbacks=[(plt.close, [], {})])
    timer.start()
    plt.show()
    ds_minimum=popen(dirname(realpath(argv[0])) + '/sortdf.py -f dataset.txt -c 3 -hdr | head -1').read().strip()
    x_min, y_min, f_min = [float(z) for z in ds_minimum.split()]
    print(f"The first element of the sorted dataset:     {x_min} {y_min:.2f} {f_min:.4f}")
    print("Analytical solution [1]:",end="")
    rprint("                     [italic]512.0 404.23 -959.6407[/italic]")
    results = dict()
    range_tuple=(r.start,r.stop-1)
    bounds = [range_tuple, range_tuple]

    results['shgo'] = optimize.shgo(eggholder, bounds, n=200, iters=5)
    print(f"Simplicial homology global optimization [2]: {results['shgo']['x'][0]:.1f} {results['shgo']['x'][1]:.2f} {results['shgo']['fun']:.4f}")

    results['DA'] = optimize.dual_annealing(eggholder, bounds, maxiter=10000)
    print(f"Dual annealing [3]:                          {results['DA']['x'][0]:.1f} {results['DA']['x'][1]:.2f} {results['DA']['fun']:.4f}")

    print(f"Difference between SHGO and DA methods: {abs(200*(results['shgo']['fun']-results['DA']['fun']))
                                                            /(results['shgo']['fun']+results['DA']['fun']):.2e} %")
    print("[1] EGGHOLDER: https://www.sfu.ca/~ssurjano/egg.html")
    print("[2] SHGO: https://link.springer.com/article/10.1007/s10898-018-0645-y")
    print("[3] DA: https://www.jstatsoft.org/article/view/v060i06")
    print("[4] SCIPY: https://docs.scipy.org/doc/scipy/tutorial/optimize.html#global-optimization")

if __name__ == "__main__":
    main()
