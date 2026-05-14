#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="logs_fcn_sweep"
mkdir -p "$LOG_DIR"

MODEL_INITS=(1 2 3 4 5)
NSIDES=(16 32 64)

MAX_PARALLEL=2

lmax_for_nside () {
    local nside="$1"

    case "$nside" in
        16) echo 40 ;;
        32) echo 80 ;;
        64) echo 160 ;;
        *)
            echo "Unsupported nside: ${nside}" >&2
            return 1
            ;;
    esac
}

run_job () {
    local model_init="$1"
    local nside="$2"
    local lmax="$3"

    local tag="init_${model_init}_nside_${nside}_lmax_${lmax}"
    local log_file="${LOG_DIR}/${tag}.log"

    echo "============================================================"
    echo "Starting job:"
    echo "  model_init = ${model_init}"
    echo "  nside      = ${nside}"
    echo "  lmax       = ${lmax}"
    echo "  log_file   = ${log_file}"
    echo "============================================================"

    {
        echo "Started: $(date)"
        echo "model_init = ${model_init}"
        echo "nside      = ${nside}"
        echo "lmax       = ${lmax}"
        echo

        python main_fcn_d2ps.py -m \
            model_init="${model_init}" \
            nside="${nside}" \
            lmax="${lmax}"

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
    for nside in "${NSIDES[@]}"; do
        lmax="$(lmax_for_nside "${nside}")"

        run_job "${model_init}" "${nside}" "${lmax}" &

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
    for nside in "${NSIDES[@]}"; do
        lmax="$(lmax_for_nside "${nside}")"

        tag="init_${model_init}_nside_${nside}_lmax_${lmax}"
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
