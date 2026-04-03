#!/usr/bin/env bash
script_path="$(dirname "$(realpath "$0")")"
name=Constraint_dora
log=${name}.log
dataset=${name}.csv.gz
if [[ $# -gt 0 ]]; then
    if [[ "-clean" == "$1" ]]; then
        rm -f $dataset 2>/dev/null
        rm -f ${name}* *.log 2>/dev/null
        exit 0
    fi
fi
name_lc="$(echo "$name" | tr '[:upper:]' '[:lower:]')"
"${script_path}/${name_lc}_dataset.py" -timeout 5
results=${name}_poly_optimization_results.txt
rm -f "$results" 2>/dev/null
smlp_args=(
    -data ${name}.csv.gz   # input CSV dataset
    -spec ${name_lc}.json     # JSON spec file
    -pref ${name}          # output file prefix
    -mode optimize         # operation mode
    -model poly_sklearn    # model type
    -epsilon 0.0000005     # convergence threshold
)

smlp "${smlp_args[@]}" >"$log" 2>&1
for var in X1 X2 Y1; do
    echo "$var = $(jq ".${var}.value_in_config" ${name}_${name}_optimization_results.json)" 2>&1 | tee -a "$results"
done
diff "$results" ${name_lc}_poly_optimization_results_expected.txt
