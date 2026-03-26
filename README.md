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
 - install Python 3.11; for example via python virtual environment [[pyenv]](https://github.com/pyenv/pyenv)
  ```
  pyenv install 3.11
  pyenv local 3.11
  ```
  
 - install smlptech package:
 
 ```pip install smlptech```

### Docker
### Sources

## Quick instructions on testing whether the tool works:
## TODO paths for regression data ? or separate download

   ```
    smlp \
    -data "data/smlp_toy_num_resp_mult" \
    -out_dir ./ -pref Test83 -mode optimize -pareto t \
    -resp y1,y2 -feat x,p1,p2 -model dt_sklearn -dt_sklearn_max_depth 15 \
    -spec smlp_toy_num_resp_mult_free_inps -data_scaler min_max \
    -beta "y1>7 and y2>6" -objv_names obj1,objv2,objv3 \
    -objv_exprs "(y1+y2)/2;y1/2-y2;y2" -epsilon 0.05 -delta_rel 0.01 \
    -save_model_config f -mrmr_pred 0 -plots f -seed 10 -log_time f \
    -spec ../specs/smlp_toy_num_resp_mult_free_inps.spec
```

This should produce ...
(better visual example; eggholder ?  )

## Tutorial

  Tutorial contains SMLP examples for black-box function optimization and Intel benchmarks for signal integrity. 


   


   