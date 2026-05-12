#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="logs_fcn_sweep"
mkdir -p "$LOG_DIR"

MODEL_INITS=(2 3 4 5)
OUT_POWERS=(1 2)

MAX_PARALLEL=2

run_job () {
    local model_init="$1"
    local out_power="$2"

    local tag="init_${model_init}_op_${out_power}"
    local log_file="${LOG_DIR}/${tag}.log"

    echo "============================================================"
    echo "Starting job:"
    echo "  model_init = ${model_init}"
    echo "  out_power  = ${out_power}"
    echo "  log_file   = ${log_file}"
    echo "============================================================"

    {
        echo "Started: $(date)"
        echo "model_init = ${model_init}"
        echo "out_power  = ${out_power}"
        echo

        python main_fcn_d2ps.py -m \
            model_init="${model_init}" \
            out_power="${out_power}"

        echo
        echo "Finished process: $(date)"
    } > "${log_file}" 2>&1

    if grep -q "Pipeline completed." "${log_file}" && \
       grep -q "Done running stage: TrainFCNModelExecutor" "${log_file}"; then
        echo "COMPLETE: ${tag}"
    else
        echo "FAILED OR INCOMPLETE: ${tag}"
        echo "  Check: ${log_file}"
    fi
}

for model_init in "${MODEL_INITS[@]}"; do
    for out_power in "${OUT_POWERS[@]}"; do

        run_job "${model_init}" "${out_power}" &

        while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
            sleep 5
        done

    done
done

wait

echo
echo "============================================================"
echo "All launched jobs have exited. Final status:"
echo "============================================================"

for model_init in "${MODEL_INITS[@]}"; do
    for out_power in "${OUT_POWERS[@]}"; do
        tag="init_${model_init}_op_${out_power}"
        log_file="${LOG_DIR}/${tag}.log"

        if grep -q "Pipeline completed." "${log_file}" && \
           grep -q "Done running stage: TrainFCNModelExecutor" "${log_file}"; then
            echo "COMPLETE: ${tag}"
        else
            echo "FAILED OR INCOMPLETE: ${tag}"
            echo "  ${log_file}"
        fi
    done
done
