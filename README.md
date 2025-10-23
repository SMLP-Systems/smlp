# SMLP Visualization Framework Extension

SMLP is a tool for optimization, synthesis and design space exploration. It is
based on statistical and machine learning techniques combined with formal verification
approaches that allows selection of optimal configurations with respect to given
constraints on the inputs and outputs of the system under consideration.

SMLP has been applied at Intel for hardware design optimization. It is a gneral
purpose optimization and verification tool applicable to any domain where ML models
can be trained on data. The supported modes in SMLP include processing data and
training models (NNs, tree-based, and polynomial models), querying data and models,
certifying and verifying assertions, synthesis and pareto-optimization of configurations,
design of experiments (to simulate systems and produce data), feature selection,
rule learning/subgroup discovery, root-cause analysis, and more.

If you want to try out SMLP on your optimization problems
and require support, please contact the developers through
the [discussion page](https://github.com/SMLP-Systems/smlp/discussions).

## Overview

The visualization framework is integrated alongside the main SMLP codebase. It allows users to generate insightful plots that illustrate how SMLP identifies stable witnesses and avoids counter-examples within the defined search space.
Coming soon: Support for NLP and LLMs in SMLP.
See [SMLP_manual_v2.pdf](./SMLP_manual_v2.pdf) for more details.

NLP:

- NLP based text classification. Applicable to spam detection, sentiment analysis, and more.
- NLP based root cause analysis: which words or collections of words are most correlative to classification decision (especially, for the positive class).

LLM:

- LLM training from scratch
- LLM finetuning
- RAG (with HuggingFace and with LangChain)

Agentic:

- SMLP Agent

## Usage

1. **Dataset Location:** Ensure your SMLP experiment datasets (the `.spec` file and the corresponding `.csv` data file) are located within the `experiment_outputs/` directory, organized into subdirectories named according to their `setno` (e.g., `experiment_outputs/set4/`, `experiment_outputs/set10/`, etc.).

2. **Configuration (Current Requirement):**
   - Navigate to the visualization code located in `src/smlp_py/ext/`.
   - Open the relevant Python file containing the `plot_exp` class (likely `plot.py` or similar).
   - In the `__init__` method of the `plot_exp` class, locate the `setno` parameter.
   - **Manually change** the value assigned to `setno` to match the dataset set number you wish to analyze (e.g., `setno='4'`, `setno='10'`).
   - _(Note: This manual step is a current requirement and may be streamlined in future versions.)_

3. **Running SMLP with Visualization:**
   - When you run SMLP after setting the correct `setno` in the code, the framework will automatically copy the specified dataset (`.spec` and `.csv`) from its subdirectory (e.g., `experiment_outputs/set4/`) into the base directory where SMLP expects to find its input files.
   - Proceed with running your SMLP experiment (e.g., optimization mode) as usual.

4. **Output Files:** During and after the SMLP run, the visualization framework will generate several output files in the main project directory (or a specified output directory), prefixed with the `setno` and experiment number (`expno`):

- **`Set{setno}_experiments.json`**: Contains comprehensive data for all experiments conducted on the specified `setno`, stored in JSON format.
- **`Set{setno}_experiments_output.ods`**: Presents key experiment results in a structured OpenDocument Spreadsheet table format for easy viewing and analysis.
- \*\*`Set{setno}_{expno}_witnesses.json`\*\*: Details information about all witnesses found during a specific experiment (`expno`) for the given `setno`.
- **`Set{setno}_{expno}_scatter_plot_predicted.html`**: An interactive HTML plot showing the performance of the trained model on the test set (predicted vs. actual values).
- \*\*`Set{setno}_{expno}_stable_x_counter.html`\*\*: _(Key Visualization)_ An interactive HTML plot overlaying the discovered stable witnesses and identified counter-examples onto the original dataset's distribution. This plot is crucial for understanding the explored and excluded regions of the solution space during the stability search.
- **`Set{setno}_{expno}_witnesses.html`**: _(Key Visualization)_ An interactive HTML plot showing all discovered witnesses plotted against the original dataset distribution. Markers indicate found witnesses; experiments yielding no witnesses will naturally have no markers shown on this plot for that specific run.
  git clone <https://github.com/fbrausse/kay.git>
  git clone <https://github.com/fbrausse/smlp.git>
  cd smlp/utils/poly
  git clone <https://github.com/smlp-systems/kay.git>
  git clone <https://github.com/smlp-systems/smlp.git>
  cd smlp/utils/poly

## Core SMLP Functionality

Apart from the added visualization outputs and the initial dataset copying mechanism, the core logic and functionality of SMLP (model training, optimization algorithms, SMT solver interaction, etc.) remain unchanged.

---

**Note:** This extension provides valuable visual feedback on the SMLP optimization process, aiding in understanding convergence behavior, the effectiveness of stability constraints, and the distribution of potential solutions. Remember to correctly set the `setno` parameter in the visualization code before running your experiments.
