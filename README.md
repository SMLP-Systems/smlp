# SMLP - Symbolic Machine Learning Prover

SMLP is a general purpose tool for verification and optimisation of systems modelled using machine learning. </br>
SMLP uses symbolic reasoning for ML model exploration and optimisation under verification and stability constraints.

<img src="https://raw.githubusercontent.com/SMLP-Systems/smlp/master/misc/smlp_overview.png" alt="SMLP Overview" class="center" width="750" height="500">


**Industry adoption:** SMLP is used at **Intel** in production for optimization of package/board layouts and signal integrity

<details>
<summary> SMLP applications in Intel and why stability is important  </summary><br>

SMLP has been successfully used at Intel to optimize package and board layouts under noisy, real‑world signal‑integrity data collected in the lab. 
Because this data is inherently noisy—and because ML models are often intentionally approximate to avoid overfitting—robustness is essential when searching for reliable optimal solutions. 
SMLP addresses this through its notion of stability, ensuring that selected optima remain valid under data and model uncertainty.
In most cases the stability radius (*) is actually as large
as 10% of the value of the variable in the configuration.
This is because the sampling error from analog equipment
can be dependent on the intended value itself.
  
 [(*)](https://ece.technion.ac.il/wp-content/uploads/2021/01/publication_617-1.pdf) The smallest perturbation 
 (measured by a norm, e.g., Chebyshev) that makes an optimal solution either non-optimal or infeasible. 

</details><br>


**[Combination of robustness and formal assurance of results validity](https://korovin.gitlab.io/pub/fmcad_bkk_2020.pdf) is a distinctive strength of SMLP, not found in other optimization or model‑analysis tools.**

SMLP exploration modes:

- optimization 
- verification
- synthesis
- query
- root cause analysis


Systems analysed by SMLP can be represented as:

- black-box functions: via sampling input -- output behaviour
  only tabular data is needed  
- explicit expressions involving polynomials
- machine learning models:
  - neural networks
  - decision trees
  - random forests
  - polynomial models


SMLP supports:
 - symbolic constrains
 - stability constraints
 - parameter optimization

<p align="center">
<img src="https://raw.githubusercontent.com/SMLP-Systems/smlp/master/misc/smlp_arch.png"  alt="SMLP Arch" class="center" width="800" height="500">
</p>

Papers:
* SMLP: Symbolic Machine Learning Prover, (CAV'24) [[pdf]](https://link.springer.com/content/pdf/10.1007/978-3-031-65627-9_11.pdf) [[bib]](https://dblp.org/rec/conf/cav/BrausseKK24.html?view=bibtex)
* Combining Constraint Solving and Bayesian Techniques for System Optimization, (IJCAI'22) [[pdf]](https://www.ijcai.org/proceedings/2022/0249.pdf) [[bib]](https://dblp.org/rec/conf/ijcai/BrausseKK22.html?view=bibtex)
* Selecting Stable Safe Configurations for Systems Modelled by Neural Networks with ReLU Activation, (FMCAD'20), [[pdf]](https://korovin.gitlab.io/pub/fmcad_bkk_2020.pdf) [[bib]](https://dblp.org/rec/conf/fmcad/BrausseKK20.html?view=bibtex)

## Installation

### pip installation (recommended)

 <details>
 <summary> MacOS </summary>
 

 - install Python 3.11; for example via Python version management [[pyenv]](https://github.com/pyenv/pyenv)
 
  ```
  pyenv install 3.11
  pyenv local 3.11
  ```
  
 - install smlptech package:
 
 ```
 pip install smlptech
 ```

</details>

<details>
 <summary> Ubuntu 24.04 </summary>
 
  * `cd scripts/venv/`
  
  * Follow: [[SMLP Installation Guide for Ubuntu-24.04]](https://github.com/SMLP-Systems/smlp/blob/master/scripts/venv/README.md)

</details>


### Docker
 <details>
 <summary> MacOS </summary>
 
  ``` docker run -it mdmitry1/python311-dev-mac:latest ```
  
  Within docker container prepend SMLP Python script with `xvfb-run`.  
  For example: 

  ```bash
  xvfb-run smlp -h
  ```

</details>

<details>
 <summary> Linux </summary>
 
  ```docker run -it mdmitry1/python311-dev:latest```
  
 Within docker container prepend SMLP Python script with `xvfb-run`.  
 For example: 

  ```bash
  xvfb-run smlp -h
 ```
 
</details>

<details>
 <summary> Linux with GUI support using VNC </summary>


Starting VNC server within container:

```
./start_vnc
```

```
scripts/bin/enter_container
```

Starting VNC server within container:

```
./start_vnc
```

Recommended VNC client: 

- Ubuntu: `remmina`
- Windows: RealVNC®
  
  Details - see [RealVNC® installation instructions](doc/RealVNC.md)

</details>


<details>
 <summary> Linux with GUI support using X11 </summary>

- Entering Docker container with X11 support on native Linux

```
scripts/bin/enter_container_x11_forwarding
```

Dependencies: `socat`

- Entering Docker container with X11 support on wslg

```
scripts/bin/enter_container_wslg
```

Dependencies: `WSL2` with `WSLG` enabled

- Installation test:
```
tests/install/test_container_install mdmitry1/python311-dev
```

</details>


### Sources
<details>
 <summary> Installation on a stock Ubuntu-22.04 </summary>
 
``` 
	sudo apt install \
		python3-pip ninja-build z3 libz3-dev libboost-python-dev texlive \
		pkg-config libgmp-dev libpython3-all-dev python-is-python3
	# get a recent version of the meson configure tool
	pip install --user meson

	# obtain sources
	git clone https://github.com/fbrausse/kay.git
	git clone https://github.com/smlp-systems/smlp.git
	cd smlp/utils/poly

	# workaround <https://bugs.launchpad.net/ubuntu/+source/swig/+bug/1746755>
	echo 'export PYTHONPATH=$HOME/.local/lib/python3/dist-packages:$PYTHONPATH' >> ~/.profile
	# get $HOME/.local/bin into PATH and get PYTHONPATH
	mkdir -p $HOME/.local/bin
	source ~/.profile

	# setup, build & install libsmlp
	meson setup -Dkay-prefix=$HOME/kay --prefix $HOME/.local build
	ninja -C build install

	# tensorflow-2.16 has a change leading to the error:
	# 'The filepath provided must end in .keras (Keras model format).'
	pip install --user \
		pandas tensorflow==2.15.1 scikit-learn pycaret seaborn \
		mrmr-selection jenkspy pysubgroup pyDOE doepy
```                

 </details>

<details>
 <summary> MacOS </summary>
   Installation instructions for Ubuntu-22.04 can be followed using `homebrew` in place of `apt`
</details>

## Quickstart

###  Problem: find minimal distance between point (2,1) and unit circle<br>
 
 <p align="left">
<img src="https://raw.githubusercontent.com/SMLP-Systems/smlp/master/misc/minimal_distance.png"  alt="Minimal Distance Problem" class="center" width="500" height="400"></p>
 
 Analytical solution for this problem is:<br>
 `
f(x*) = 6 - 2√5 ≈ 1.527864`, where `x* = (2/√5,1/√5) ≈ (0.894427, 0.447214)`
 <br>

Let's solve this problem using SMLP.

Download and unzip [quickstart.zip](https://raw.githubusercontent.com/SMLP-Systems/smlp/smlp-quickstart-v2/misc/quickstart.zip)
(or if you cloned smlp cd to quickstart)

Let's treat this problem as black-box function optimization.

<details>

<summary> Step 1: Generate samples of the distance function from the point (2,1) (for simplicity we use square of the distance as this does not change the optimum point):
</summary>

Run:

```
./constraint_dora_dataset.py
```

This should generate `Constraint_dora.csv.gz`, inside `Constraint_dora.csv` we have:

```
X1,X2,Y1
-1.5,-1.5,18.5
-1.495995995995996,-1.5,18.471988004020037
-1.491991991991992,-1.5,18.4440080721362
-1.487987987987988,-1.5,18.41606020434849
......
```
</details>

<details>

<summary>
Step 2: Create specification file (or use provided `constraint_dora.json`) where we specify types and ranges of variables and that the solution should be constrained to the unit circle:
</summary>


```
{
  "version": "1.2",
  "variables": [
    {"label":"X1", "interface":"knob", "type":"real", "range":[-1.5,2.5], "rad-abs": 0.0},
    {"label":"X2", "interface":"knob", "type":"real", "range":[-1.5,2.0], "rad-abs": 0.0},
    {"label":"Y1", "interface":"output", "type":"real"}
  ],
  "beta": "X1*X1+X2*X2<=1",
  "objectives": {
    "objective1": "-Y1"
  }
}
```
   <u>Legend:</u><br> 

```
   X1 - first controllable variable
   X2 - second controlllable variable
   Y1 - output function
   rad-abs - sensitivity radius. 
             Zero radius means that solution sensitivity check is skipped
   beta - constraint depending on controllable variables
   objective1 - optimization goal
```

Note SMLP by default maximizes the objective function so we use `-Y1` as the objective function.

</details>

<details>
<summary>
Step 3: Run SMLP on data file and specification file:

</summary>

```
smlp -data Constraint_dora.csv.gz -spec ./constraint_dora.json -pref results/Constraint_dora -mode optimize -model poly_sklearn -epsilon 0.0000005
```

SMLP command line arguments:<br>

   ```
    -data ${name}.csv.gz                  # input CSV dataset
    -spec ${script_path}/${name_lc}.json  # JSON spec file
    -pref ${name}                         # output file prefix
    -mode optimize                        # operation mode
    -model poly_sklearn                   # model type
    -epsilon 0.0000005                    # convergence threshold
```


3 graphs will pop-up which show quality of the generated model in train/validation/test dataset split, (these need to be closed to proceed). <br>
The generated results can be found in `results/` folder.  <br>

`results/Constraint_dora_Constraint_dora_optimization_results.csv` contains the generated solution:

```
X1 = 0.89453125
X2 = 0.4470043182373047
Y1 = 1.5278653812777188
```

Solution found by SMLP corresponds to the analytical solution (`constraint_dora_poly_optimization_results_expected.txt`) with the specified precision:

```
X1 = 0.89453125
X2 = 0.4470043182373047
Y1 = 1.5278653812779421
```
</details>

Steps 1 - 3 are wrapped in a script: `./quickstart.sh`

<details>
<summary> Step 4: As an example, let's modify the problem in order to get solution in rational numbers.</summary>
<br>
  Let's change circle radius to 2/√5, so squared radius will be 4/5.<br>
  In order to do this, edit specification file `constraint_dora.json`  and change the right side of the inequality in the constraint to be 4/5:
  
    `"beta": "X1*X1+X2*X2<=4/5,"`
 
  Run the script from current directory 

```bash
./quickstart.sh
```

Expected SMLP results are within 0.03% accuracy for `f(x*)` and `x*`:
```bash 
Working directory: <current_directory_realpath>/quickstart/Constraint_dora_results_<timestamp>
X1 = 0.800048828125
X2 = 0.3999021053314209
Y1 = 1.8000002980730385
```

 [Analytical solution](https://www.wolframalpha.com/input?i=Minimize%3A+f%28x1%2C+x2%29+%3D+%28x1+-+2%29%5E2+%2B+%28x2+-+1%29%5E2+subject+to+x1%5E2+%2B+x2%5E2+-+4%2F5+%3C%3D+0) for this modified problem:<br>
 
 `f(x*) = 9/5 = 1.8`, where `x* = (4/5,2/5) = (0.8, 0.4)`

</details>


## [Tutorial](https://github.com/SMLP-Systems/smlp/tree/master/tutorial)

   - Black-box optimization Eggholder Function
   - Constrained DORA (Distance to Optimal with Radial Adjustment)
   - Binh and Korn (BNH) Multi-Objective Problem
   - Intel Signal Integrity domain example



## [Manual](https://github.com/SMLP-Systems/smlp/blob/master/doc/smlp_manual.pdf)

  
### Comments/Feedback/Discussions: [[GitHub]](https://github.com/SMLP-Systems/smlp/) or [[Zulip Chat]](https://smlp.zulipchat.com)


### Coming soon:
<details>
<summary> NLP, LLM, Agentic</summary>

Current development is in PR [#21](https://github.com/SMLP-Systems/smlp/pull/21)  </br>
See [Extended Manual](https://raw.githubusercontent.com/SMLP-Systems/smlp/nlp_text.rebased/doc/smlp_manual_extended.pdf) for details.

NLP:
 - NLP based text classification. Applicable to spam detection, sentiment analysis, and more.
 - NLP based root cause analysis: which words or collections of words are most correlative to classification decision (especially, for the positive class).

LLM:
- LLM training from scratch
- LLM finetuning
- RAG (with HuggingFace and with LangChain)

Agentic:
- SMLP Agent

</details>
