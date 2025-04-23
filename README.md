# SMLP Visualization Framework Extension

This document describes an extension to the SMLP (Statistical Machine Learning Platform) project, incorporating a visualization framework designed to plot and analyze the witness exploration process during optimization runs.

## Overview

The visualization framework is integrated alongside the main SMLP codebase. It allows users to generate insightful plots that illustrate how SMLP identifies stable witnesses and avoids counter-examples within the defined search space.

## Usage

1.  **Dataset Location:** Ensure your SMLP experiment datasets (the `.spec` file and the corresponding `.csv` data file) are located within the `experiment_outputs/` directory, organized into subdirectories named according to their `setno` (e.g., `experiment_outputs/set4/`, `experiment_outputs/set10/`, etc.).

2.  **Configuration (Current Requirement):**
    *   Navigate to the visualization code located in `src/smlp_py/ext/`.
    *   Open the relevant Python file containing the `plot_exp` class (likely `plot.py` or similar).
    *   In the `__init__` method of the `plot_exp` class, locate the `setno` parameter.
    *   **Manually change** the value assigned to `setno` to match the dataset set number you wish to analyze (e.g., `setno='4'`, `setno='10'`).
    *   *(Note: This manual step is a current requirement and may be streamlined in future versions.)*

3.  **Running SMLP with Visualization:**
    *   When you run SMLP after setting the correct `setno` in the code, the framework will automatically copy the specified dataset (`.spec` and `.csv`) from its subdirectory (e.g., `experiment_outputs/set4/`) into the base directory where SMLP expects to find its input files.
    *   Proceed with running your SMLP experiment (e.g., optimization mode) as usual.

4.  **Output Files:** During and after the SMLP run, the visualization framework will generate several output files in the main project directory (or a specified output directory), prefixed with the `setno` and experiment number (`expno`):

    *   **`Set{setno}_experiments.json`**: Contains comprehensive data for all experiments conducted on the specified `setno`, stored in JSON format.
    *   **`Set{setno}_experiments_output.ods`**: Presents key experiment results in a structured OpenDocument Spreadsheet table format for easy viewing and analysis.
    *   **`Set{setno}_{expno}_witnesses.json`**: Details information about all witnesses found during a specific experiment (`expno`) for the given `setno`.
    *   **`Set{setno}_{expno}_scatter_plot_predicted.html`**: An interactive HTML plot showing the performance of the trained model on the test set (predicted vs. actual values).
    *   **`Set{setno}_{expno}_stable_x_counter.html`**: *(Key Visualization)* An interactive HTML plot overlaying the discovered stable witnesses and identified counter-examples onto the original dataset's distribution. This plot is crucial for understanding the explored and excluded regions of the solution space during the stability search.
    *   **`Set{setno}_{expno}_witnesses.html`**: *(Key Visualization)* An interactive HTML plot showing all discovered witnesses plotted against the original dataset distribution. Markers indicate found witnesses; experiments yielding no witnesses will naturally have no markers shown on this plot for that specific run.

## Core SMLP Functionality

Apart from the added visualization outputs and the initial dataset copying mechanism, the core logic and functionality of SMLP (model training, optimization algorithms, SMT solver interaction, etc.) remain unchanged.

---

**Note:** This extension provides valuable visual feedback on the SMLP optimization process, aiding in understanding convergence behavior, the effectiveness of stability constraints, and the distribution of potential solutions. Remember to correctly set the `setno` parameter in the visualization code before running your experiments.
