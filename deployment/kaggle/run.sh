#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash deployment/kaggle/run.sh --mode MODE [options]

Modes:
  setup    Install DE dependencies in the current Python environment.
  smoke    Check the DE pipeline with up to 5 validation samples and 1 generation.
  evolve   Run DE, resuming a matching checkpoint if found.

Options for smoke and evolve:
  --data-root PATH    Attached Parquet dataset root (default: /kaggle/input/vnmedical-bv175).
  --work-root PATH    Write to PATH/runs/de or PATH/runs/de_smoke (default: /kaggle/working).
  --config PATH       DE YAML configuration (default: configs/de.yaml).
  --set KEY VALUE     Override a YAML setting. May be repeated.
  -h, --help          Show this help.

Relative paths are resolved from the repository root. Use PYTHON_BIN to select Python.
EOF
}

fail() {
    printf 'Error: %s\n' "$1" >&2
    exit 2
}

mode=""
data_root="/kaggle/input/vnmedical-bv175"
work_root="/kaggle/working"
config="configs/de.yaml"
overrides=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --mode|--data-root|--work-root|--config)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires a value."
            case "$1" in
                --mode) mode="$2" ;;
                --data-root) data_root="$2" ;;
                --work-root) work_root="$2" ;;
                --config) config="$2" ;;
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

case "$mode" in
    setup)
        exec "$python_bin" -m pip install -r deployment/kaggle/requirements.txt
        ;;
    smoke|evolve)
        [[ -f "$config" ]] || fail "Config file not found: $config"
        args=(--config "$config"
              --set data.dataset_dir "$data_root"
              --set de.cache_dir "${work_root%/}/runs/de")
        if [[ "$mode" == smoke ]]; then
            args+=(--smoke)
        fi
        export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
        exec "$python_bin" -u -m IT4906.de.run_de "${args[@]}" ${overrides[@]+"${overrides[@]}"}
        ;;
    *) fail "Use --mode setup, --mode smoke, or --mode evolve. See --help." ;;
esac
