#!/usr/bin/python3.12
from pandas import read_csv
from matplotlib import pyplot as plt
from sys import argv
from math import inf

def plot_results(csv: str = "results.csv", timeout: float =inf) -> int:
    df = read_csv(csv,sep=',')
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(df['F1'], df['F2'], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    plt.xlabel('F1', fontsize=12)
    plt.ylabel('F2', fontsize=12)
    plt.title('Scatter Plot of Objective Functions F1 vs F2', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    if not inf == timeout:
        timer = fig.canvas.new_timer(interval=int(timeout), callbacks=[(plt.close, [], {})])
        timer.start()
    plt.show()
    return 0

if __name__ == "__main__":
    csv = "results.csv" if len(argv) < 2 else argv[1]
    timeout = inf if len(argv) < 3 else argv[2]
    plot_results(csv, timeout)
