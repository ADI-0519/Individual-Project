#!/usr/bin/env bash
#SBATCH --job-name=latent-probe-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=latent-probe-%j.out

# Planning defaults above are not measured resource requirements. Override with
# sbatch options after a one-example memory/time check. The partition and any
# account/constraint must match the actual CREATE allocation; for example:
#   sbatch --partition=gpu scripts/create_gpu_job.sh sender
# Load the lab-approved modules and select an installed Python environment before
# submission. This script intentionally does not guess CUDA/module versions.
set -euo pipefail

: "${LATENT_PROJECT_DIR:?Set LATENT_PROJECT_DIR to the absolute repository path}"
: "${LATENT_OUTPUT:?Set LATENT_OUTPUT to a new sender-run output path or the existing run for recipients}"

LATENT_PYTHON="${LATENT_PYTHON:-python}"
LATENT_CONFIG="${LATENT_CONFIG:-configs/llama8b_4bit.json}"
LATENT_SCENARIOS="${LATENT_SCENARIOS:-data/pilot_scenarios.jsonl}"
LATENT_STAGE="${1:-sender}"
LATENT_RESUME="${LATENT_RESUME:-false}"

latent_extra_args=()
case "$LATENT_RESUME" in
    true) latent_extra_args+=(--resume) ;;
    false) ;;
    *)
        printf '%s\n' 'LATENT_RESUME must be true or false.' >&2
        exit 2
        ;;
esac

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' 'Submit this script with sbatch to an approved GPU partition.' >&2
    exit 2
fi
if [[ ! -d "$LATENT_PROJECT_DIR" ]]; then
    printf '%s\n' 'LATENT_PROJECT_DIR is not an accessible directory.' >&2
    exit 2
fi
cd "$LATENT_PROJECT_DIR"

if ! command -v "$LATENT_PYTHON" >/dev/null 2>&1; then
    printf '%s\n' 'LATENT_PYTHON does not identify an available Python executable.' >&2
    exit 2
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

case "$LATENT_STAGE" in
    sender)
        exec "$LATENT_PYTHON" -m latent_probes run \
            --config "$LATENT_CONFIG" \
            --scenarios "$LATENT_SCENARIOS" \
            --output "$LATENT_OUTPUT" \
            "${latent_extra_args[@]}"
        ;;
    recipients)
        if [[ -n "${LATENT_RECIPIENT_CONFIG:-}" ]]; then
            latent_extra_args+=(--config "$LATENT_RECIPIENT_CONFIG")
        fi
        exec "$LATENT_PYTHON" -m latent_probes replay-recipients \
            --run "$LATENT_OUTPUT" \
            "${latent_extra_args[@]}"
        ;;
    *)
        printf '%s\n' 'Stage must be sender or recipients.' >&2
        exit 2
        ;;
esac
