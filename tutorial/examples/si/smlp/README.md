### 1. Pull the image
```bash
docker pull mdmitry1/python311-dev:latest
```
### 2. Run the container
```bash
docker run -it mdmitry1/python311-dev:latest
```
### 3. In the container
```bash
cd smlp
git checkout remotes/origin/poly_pareto tutorial/examples/si/smlp
xvfb-run tutorial/examples/si/smlp/run_si_test_nosplit
```
### 4. Runtime on Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz in Docker environment, 4 Gb allocated for Docker VM
1577.631u 16.508s 26:43.48 99.4%	0+0k 1064312+417720io 3400pf+0w

### 5. Validation
```bash
diff no_split_s2_tx_piv_anonym_optimization_results.json tutorial/examples/si/smlp/NO_SPLIT_S2_TX_PIV_ANONYM_OPTIMIZATION_RESULTS.json
diff no_split_s2_tx_piv_anonym_optimization_results.csv tutorial/examples/si/smlp/NO_SPLIT_S2_TX_PIV_ANONYM_OPTIMIZATION_RESULTS.csv
```
