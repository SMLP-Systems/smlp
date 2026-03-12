#!/usr/bin/env bash
script_path=$(dirname "$(realpath "$0")")
test=BNH
test_lc=$(echo "$test" | tr '[A-Z]' '[a-z]')
gzipped_dataset="${test_lc}.csv.gz"

if [[ $# -gt 0 ]]; then
    if [[ "-clean" == "$1" ]]; then
        rm -f ${test}* *.log > /dev/null 2>&1
        rm -f "$gzipped_dataset" > /dev/null 2>&1
        exit 0
    fi
fi

${script_path}/bnh_dataset.py

csv="${test}.csv"
json="${script_path}/${test_lc}_w.json"

gunzip -c "$gzipped_dataset" > "$csv"

o1="(-F1)"
o2="(-(F1*0.8+F2*0.2))"
o3="(-(F1*0.6+F2*0.4))"
o4="(-(F1*0.4+F2*0.6))"
o5="(-(F1*0.2+F2*0.8))"
o6="(-F2)"

objs=("$o1" "$o2" "$o3" "$o4" "$o5" "$o6")

n=1
for obj in "$o1" "$o2" "$o3" "$o4" "$o5" "$o6"; do
    python3.11 "$(realpath "$0" | xargs dirname | xargs dirname | xargs dirname | xargs dirname | xargs dirname)/src/run_smlp.py" \
        -data "$csv" \
        -spec "$json" \
        -out_dir ./ \
        -pref "$test" \
        -mode optimize \
        -pareto f \
        -opt_strategy eager \
        -resp F1,F2 \
        -feat X1,X2 \
        -objv_exprs "$obj" \
        -model poly_sklearn \
        -tree_encoding flat \
        -compress_rules t \
        -mrmr_pred 0 \
        -epsilon 0.000005 \
        -delta_rel 0.05 \
        -save_model t \
        -model_name "${csv%.*}_model" \
        -save_model_config t \
        -plots f \
        -pred_plots f \
        -resp_plots f \
        -seed 10 \
        -log_time f > "$(basename "$0").log" 2>&1
    mv -f "BNH_BNH_optimization_results.json" "${test_lc}_p${n}_optimization_results.json"
    (( n++ ))
done
