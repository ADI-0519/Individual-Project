#!/usr/bin/env bash
#SBATCH --job-name=apollo-extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=apollo-extract-%j.log

# UNTESTED ON CREATE. These are initial resource caps, not measured requirements.
# Select an approved GPU partition/account/constraint at submission, for example:
#   sbatch --partition=gpu scripts/create_apollo_gpu_job.sh --config ... --examples ... --output ...
# Use an existing ARC-supported environment and any exact module setup it needs.
# This template installs nothing and assumes no CUDA module or driver version.
set -euo pipefail

: "${LATENT_PROJECT_DIR:?Set LATENT_PROJECT_DIR to the absolute repository root}"
: "${PYTHON_EXE:?Set PYTHON_EXE to the absolute Python path in an existing supported environment}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' 'Submit this template through Slurm to an approved GPU partition.' >&2
    exit 2
fi
if [[ "$LATENT_PROJECT_DIR" != /* || ! -d "$LATENT_PROJECT_DIR" ]]; then
    printf '%s\n' 'LATENT_PROJECT_DIR must be an accessible absolute repository path.' >&2
    exit 2
fi
if [[ "$PYTHON_EXE" != /* || ! -f "$PYTHON_EXE" || ! -x "$PYTHON_EXE" ]]; then
    printf '%s\n' 'PYTHON_EXE must identify an existing executable Python by absolute path.' >&2
    exit 2
fi
if [[ $# -eq 0 ]]; then
    printf '%s\n' 'Provide Apollo extract arguments: --config ... --examples ... --output ...' >&2
    exit 2
fi

cd -- "$LATENT_PROJECT_DIR"
if [[ ! -f pyproject.toml || ! -f src/latent_probes/apollo.py || ! -f src/latent_probes/backends.py ]]; then
    printf '%s\n' 'The working directory is not the expected latent-probes repository root.' >&2
    exit 2
fi

# Execute this checkout's source using dependencies from the selected environment.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# doctor can return a valid report when CUDA is absent. Check its report fields,
# including the actual matrix-kernel result, instead of trusting CLI exit zero.
"$PYTHON_EXE" - <<'PY'
import json
import sys
from pathlib import Path

if sys.version_info < (3, 11):
    raise SystemExit("This project requires Python 3.11 or newer.")

import latent_probes
from latent_probes.backends import doctor

if Path(latent_probes.__file__).resolve().parent != Path("src/latent_probes").resolve():
    raise SystemExit("The selected interpreter did not import this repository's source.")

report = doctor()
print(json.dumps(report, indent=2), flush=True)
cuda = report.get("cuda", {})
if cuda.get("available") is not True or cuda.get("kernel_check_passed") is not True:
    raise SystemExit("CUDA availability and the actual GPU kernel check must both pass.")
PY

exec "$PYTHON_EXE" -m latent_probes apollo extract "$@"
