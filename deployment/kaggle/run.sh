#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash deployment/kaggle/run.sh --mode MODE [options]

Modes:
  setup    Install DE dependencies in the current Python environment.
  smoke    Check the DE pipeline with up to 5 validation samples and 1 generation.
  evolve   Run DE, resuming a matching checkpoint if found.
  evaluate Evaluate the saved best prompt on the held-out test split.
  push-results Upload a results folder to the IT4906-TTTH dataset on Hugging Face.

Options for smoke and evolve:
  --target-gpus N     Replicate Target LLM on 1 or 2 CUDA GPUs (default: 1).
  --val-num-samples N Use the first N validation samples (config default: 200, smoke cap: 5).
  --data-root PATH    Use attached Parquet files instead of streaming from Hugging Face.
  --work-root PATH    Write to PATH/runs/de or PATH/runs/de_smoke (default: /kaggle/working).
  --config PATH       DE YAML configuration (default: configs/de.yaml).
  --init-mode MODE    Population initialization: human (default), gpt, random, top, bottom.
  --evolution-provider PROVIDER
                      Evolution LLM provider: openai, openrouter, local.
  --evolution-model-id ID
                      Override the model ID for the Evolution LLM.
  --set KEY VALUE     Override a YAML setting. May be repeated.
  -h, --help          Show this help.

Options for evaluate:
  --target-gpus N     Override the target GPU count saved with the run.
  --work-root PATH    Find the run at PATH/runs/de (default: /kaggle/working).
  --run-dir PATH      Read best_prompt.json and run_config.json from PATH instead.
  --data-root PATH    Override the saved Parquet dataset root for test evaluation.
  --config PATH       Fallback config for old results with only best_prompt.json.
  --eval-num-samples N
                      Evaluate the first N test samples. Defaults to the full test split.

Options for push-results (requires HF_TOKEN with write access):
  --local-folder PATH Local directory to upload.
  --hf-folder PATH    Destination subfolder inside vohuutridung/IT4906-TTTH.

Relative paths are resolved from the repository root. Use PYTHON_BIN to select Python.
EOF
}

fail() {
    printf 'Error: %s\n' "$1" >&2
    exit 2
}

mode=""
data_root=""
work_root="/kaggle/working"
config="configs/de.yaml"
run_dir=""
eval_num_samples=""
val_num_samples=""
target_gpus=""
local_folder=""
hf_folder=""
overrides=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --mode|--data-root|--work-root|--config|--run-dir|--eval-num-samples|--val-num-samples|--target-gpus|--local-folder|--hf-folder|--init-mode|--evolution-provider|--evolution-model-id)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires a value."
            case "$1" in
                --mode) mode="$2" ;;
                --data-root) data_root="$2" ;;
                --work-root) work_root="$2" ;;
                --config) config="$2" ;;
                --run-dir) run_dir="$2" ;;
                --eval-num-samples) eval_num_samples="$2" ;;
                --val-num-samples) val_num_samples="$2" ;;
                --target-gpus) target_gpus="$2" ;;
                --local-folder) local_folder="$2" ;;
                --hf-folder) hf_folder="$2" ;;
                --init-mode) overrides+=(--set de.initial_mode "$2") ;;
                --evolution-provider|--evolution-model-id) overrides+=("$1" "$2") ;;
            esac
            shift 2
            ;;
        --set)
            [[ $# -ge 3 && -n "$2" && "$2" != --* && "$3" != --* ]] \
                || fail "--set requires a key and value."
            overrides+=(--set "$2" "$3")
            shift 3
            ;;
        *) fail "Unknown argument '$1'. See --help." ;;
    esac
done

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python3}"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$mode" != push-results && ( -n "$local_folder" || -n "$hf_folder" ) ]]; then
    fail "--local-folder and --hf-folder are for push-results."
fi
if [[ -n "$val_num_samples" && "$mode" != smoke && "$mode" != evolve ]]; then
    fail "--val-num-samples is for smoke and evolve."
fi

target_args=()
if [[ -n "$target_gpus" ]]; then
    [[ "$target_gpus" == 1 || "$target_gpus" == 2 ]] || fail "--target-gpus must be 1 or 2."
    target_args=(--target-gpus "$target_gpus")
fi

case "$mode" in
    setup)
        exec "$python_bin" -m pip install -r deployment/kaggle/requirements.txt
        ;;
    smoke|evolve)
        [[ -z "$run_dir" && -z "$eval_num_samples" ]] || fail "--run-dir and --eval-num-samples are for evaluate."
        [[ -f "$config" ]] || fail "Config file not found: $config"
        args=(--config "$config" --set de.cache_dir "${work_root%/}/runs/de")
        if [[ -n "$data_root" ]]; then
            args+=(--set data.dataset_dir "$data_root")
        fi
        if [[ -n "$val_num_samples" ]]; then
            args+=(--val-num-samples "$val_num_samples")
        fi
        if [[ "$mode" == smoke ]]; then
            args+=(--smoke)
        fi
        exec "$python_bin" -u -m IT4906.de.run_de "${args[@]}" ${overrides[@]+"${overrides[@]}"} ${target_args[@]+"${target_args[@]}"}
        ;;
    evaluate)
        [[ -z "${overrides[*]:-}" ]] || fail "evaluate uses the saved config. See --help for evaluation options."
        args=(--run-dir "${run_dir:-${work_root%/}/runs/de}" --config "$config")
        if [[ -n "$data_root" ]]; then
            args+=(--data-root "$data_root")
        fi
        if [[ -n "$eval_num_samples" ]]; then
            args+=(--num-samples "$eval_num_samples")
        fi
        exec "$python_bin" -u -m IT4906.de.evaluate_de "${args[@]}" ${target_args[@]+"${target_args[@]}"}
        ;;
    push-results)
        [[ -n "$local_folder" && -n "$hf_folder" ]] || fail "push-results requires --local-folder and --hf-folder."
        exec "$python_bin" -u -m IT4906.de.push_results --local-folder "$local_folder" --hf-folder "$hf_folder"
        ;;
    *) fail "Use --mode setup, smoke, evolve, evaluate, or push-results. See --help." ;;
esac
