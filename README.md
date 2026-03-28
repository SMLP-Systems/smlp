# SMLP -- Symbolic Machine Learning Prover

<img src="https://raw.githubusercontent.com/SMLP-Systems/smlp/master/misc/smlp_overview.png"  alt="SMLP Overview" class="center" width="800" height="500">

#### Success story: used at Intel for optimization of package/board layouts and signal integrity


SMLP is a general purpose tool for verification and optimisation of systems modelled using machine learning.  
SMLP uses symbolic reasoning for ML model exploration and optimisation under verification and stability constraints.


SMLP modes:

- optmization 
- verification
- synthesis
- exploration 
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

<img src="https://raw.githubusercontent.com/SMLP-Systems/smlp/master/misc/smlp_arch.png"  alt="SMLP Arch" class="center" width="800" height="500">

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
