#!/usr/bin/python3.13
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sys import argv
from math import inf

# Read the data files
X1 = pd.read_csv('bnh_pareto_X1.txt', header=None, names=['X1'])
X2 = pd.read_csv('bnh_pareto_X2.txt', header=None, names=['X2'])
F1 = pd.read_csv('bnh_pareto_F1.txt', header=None, names=['F1'])
F2 = pd.read_csv('bnh_pareto_F2.txt', header=None, names=['F2'])

# Combine X1 and X2
df_X = pd.concat([X1, X2], axis=1)

# Combine F1 and F2
df_F = pd.concat([F1, F2], axis=1)

# Create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot X1 vs X2
sns.scatterplot(data=df_X, x='X1', y='X2', ax=ax1)
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_title('Pareto Front in Decision Space')
ax1.grid(True)

# Plot F1 vs F2
sns.scatterplot(data=df_F, x='F1', y='F2', ax=ax2)
ax2.set_xlabel('F1')
ax2.set_ylabel('F2')
ax2.set_title('Pareto Front in Objective Space')
ax2.grid(True)

plt.tight_layout()
timeout = inf
if len(argv) > 2:
    if '-timeout' == argv[1]:
        timeout = int(argv[2]) 
if not inf == timeout:
    timer = fig.canvas.new_timer(interval=timeout*1000, callbacks=[(plt.close, [], {})])
    timer.start()
plt.show()
