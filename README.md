# SMLP -- Symbolic Machine Learning Prover

SMLP is a tool for verification and optimisation of systems modelled using machine learning.

SMLP uses symbolic reasoning for ML model exploration and optimisation under verification and stability constraints.

<img src="./misc/smlp_arch.png"  alt="SMLP" class="center" width="800" height="500">


SMLP modes:

- optmization 
- verification
- synthesis
- exploration 
- root cause analysis


Systems can be represented as:

- black-box functions: via sampling input -- output behaviour
  only tabular data is needed  
- explicit expressions involving polynomials, trigonometric functions
- machine learning models:
  - neural networks
  - decision trees
  - random forests
  - polynomial models


SMLP supports:
 - symbolic constrains
 - stability constraints
 - parameter optimization


Papers:
* SMLP: Symbolic Machine Learning Prover, (CAV'24) [[pdf]](https://link.springer.com/content/pdf/10.1007/978-3-031-65627-9_11.pdf) [[bib]](https://dblp.org/rec/conf/cav/BrausseKK24.html?view=bibtex)
* Combining Constraint Solving and Bayesian Techniques for System Optimization, (IJCAI'22) [[pdf]](https://www.ijcai.org/proceedings/2022/0249.pdf) [[bib]](https://dblp.org/rec/conf/ijcai/BrausseKK22.html?view=bibtex)
* Selecting Stable Safe Configurations for Systems Modelled by Neural Networks with ReLU Activation, (FMCAD'20), [[pdf]](https://korovin.gitlab.io/pub/fmcad_bkk_2020.pdf) [[bib]](https://dblp.org/rec/conf/fmcad/BrausseKK20.html?view=bibtex)

## Installation

### pip installation (recommended)
 - install python 3.11
 <details>
  Example:   Install python virtual environment [[pyenv]](https://github.com/pyenv/pyenv)
  ```
  pyenv install 3.11
  pyenv local 3.11
  ```
 </details>
 - `pip install smlptech`

### Docker
### Sources

