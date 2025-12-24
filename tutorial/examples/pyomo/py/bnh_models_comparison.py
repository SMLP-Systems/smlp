#!/usr/bin/python3.12
from matplotlib import pyplot as plt
from pandas import read_csv
from os.path import basename
from sys import argv
from math import inf
from base64 import b64encode
from hashlib import sha256

# Get all CSV files in current directory (or specify your path)
csv_files = ['pareto_front_analytical_expected.csv',
             'pareto_front_results_dt_expected.csv',
             'pareto_front_results_tab_expected.csv']

def main(rootpath: str = ".", timeout: float=5000, 
         csv_files: list[str] = 
             ['pareto_front_analytical_expected.csv',
             'pareto_front_results_dt_expected.csv',
             'pareto_front_results_tab_expected.csv']
          ) -> int:
  # Create figure and axis
  fig, ax = plt.subplots(figsize=(10, 6))
  
  # Define colors for different files
  colors = ['#2563eb', '#dc2626', '#16a34a', '#9333ea', 
            '#ea580c', '#0891b2', '#db2777', '#65a30d']
  
  columns_to_keep = ['F1', 'F2']
  
  # Plot each CSV file
  for idx, file in enumerate(csv_files):
      # Read CSV file (assumes 2 columns, no header)
      data = read_csv(file, sep=',')[columns_to_keep]
      
      # Get color for this file
      color = colors[idx % len(colors)]
      
      # Get filename without path
      filename = basename(file)
      
      # Plot line and points
      ax.plot(data['F1'], data['F2'], 
              color=color, 
              linewidth=2, 
              marker='o', 
              markersize=5,
              label=filename)
  
  # Customize plot
  ax.set_xlabel('F1', fontsize=12)
  ax.set_ylabel('F2', fontsize=12)
  ax.set_title('BNH Pareto front using NSGA2 algorithm', fontsize=14, fontweight='bold')
  ax.grid(True, alpha=0.3)
  ax.legend(loc='best', framealpha=0.9)
  
  # Adjust layout and display
  plt.tight_layout()
  png_file=rootpath + '/bnh_models_comparison.png'
  plt.savefig(png_file, dpi=300, bbox_inches='tight')
  if not inf == timeout:
     timer = fig.canvas.new_timer(interval=timeout, callbacks=[(plt.close, [], {})])
     timer.start()
  plt.show()
  with open(png_file, "rb") as image_file:
      image_data = image_file.read()
  encoded_image = b64encode(image_data)
  return sha256(encoded_image).hexdigest()

if __name__ == "__main__":
    rootpath = "." if len(argv) < 2 else argv[1]
    timeout = inf if len(argv) < 3 else argv[2]
    if len(argv) > 3:
        print(main(rootpath,timeout,argv[3:]))
    else:
        print(main(rootpath,timeout))

