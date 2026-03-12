#!/usr/bin/env bash
# run.sh – entry point for the AI-Powered Phishing URL Detection System
#
# Usage:
#   ./run.sh train   [options]
#   ./run.sh predict --url <URL>
#   ./run.sh predict --file <file>
#   ./run.sh explain [--url <URL>]
#
# Run `./run.sh <command> --help` for full option details.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"

# Add src/ to PYTHONPATH so that `phishdet` is importable without installing.
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

# Train a fresh model if the default model does not yet exist and no explicit
# subcommand was provided.
DEFAULT_MODEL="${SCRIPT_DIR}/models/trained_model.joblib"
if [[ "${1:-}" != "train" && ! -f "${DEFAULT_MODEL}" ]]; then
    echo "[run.sh] No trained model found at ${DEFAULT_MODEL}."
    echo "[run.sh] Running 'train' first …"
    python -m phishdet.cli train
fi

exec python -m phishdet.cli "$@"
