#!/bin/bash
# One-command install of the py-chargemol development conda environment: creates
# or updates the environment from environment-dev.yml, which installs py-chargemol
# in editable mode with its quality and test tools, then verifies the result.
# Run from anywhere:
#
#   bash build_tools/conda_install.sh
#
# An existing environment is updated in place instead of deleted and recreated.
# Set ENV_NAME to use a differently named environment:
#
#   ENV_NAME=py-chargemol-test bash build_tools/conda_install.sh
#
# Exit immediately on error and fail pipelines cleanly, so a broken install does
# not fall through to the later steps and report success.
set -eo pipefail

ENV_NAME="${ENV_NAME:-py-chargemol-dev}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required; install it or use 'python -m pip install .' instead." >&2
    exit 1
fi

echo "=== Initializing Conda Environment ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
# -n overrides the name pinned inside environment-dev.yml, so ENV_NAME works.
conda env update -n "${ENV_NAME}" -f "${SCRIPT_DIR}/environment-dev.yml"
conda activate "${ENV_NAME}"

echo "=== Verifying Installation ==="
cd "${REPO_DIR}"
python -c "import py_chargemol"
py-chargemol --version
echo "py-chargemol: OK"

echo "=== Install Complete! ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo "Then run the checks with: prek run --all-files && mypy && pytest --cov"
