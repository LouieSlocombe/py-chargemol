#!/bin/bash
# One-command install of the py-chargemol development conda environment: creates
# the environment from environment-dev.yml, which installs py-chargemol in editable
# mode with its quality and test tools, then verifies the result. Run from anywhere:
#
#   bash build_tools/conda_install.sh
#
# WARNING: the target environment (default: py-chargemol-dev) is REMOVED and
# recreated from scratch on every run. Set ENV_NAME to install into a differently
# named environment instead:
#
#   ENV_NAME=py-chargemol-test bash build_tools/conda_install.sh
#
# Exit immediately on error and fail pipelines cleanly, so a broken install does
# not fall through to the later steps and report success.
set -eo pipefail

ENV_NAME="${ENV_NAME:-py-chargemol-dev}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Initializing Conda Environment ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
# conda refuses to remove the active environment, so drop back to base first
# (covers running this script from inside an activated ${ENV_NAME}).
conda activate base
conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true
# -n overrides the name pinned inside environment-dev.yml, so ENV_NAME works.
conda env create -n "${ENV_NAME}" -f "${SCRIPT_DIR}/environment-dev.yml"
conda activate "${ENV_NAME}"

echo "=== Verifying Installation ==="
cd "${REPO_DIR}"
python -c "import py_chargemol"
py-chargemol --version
echo "py-chargemol: OK"

echo "=== Install Complete! ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo "Then run the checks with: prek run --all-files && mypy && pytest"
