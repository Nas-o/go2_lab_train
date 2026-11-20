#!/usr/bin/env bash
#
# Copyright (c) 2026 Tianyang Wu
# SPDX-License-Identifier: MIT

set -euo pipefail

GO2_RL_LAB_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_exe="${PYTHON_EXE:-python}"
go2_rl_lab_path="${GO2_RL_LAB_PATH}/source/go2_rl_lab"

ensure_optional_dependencies() {
    "${python_exe}" "${GO2_RL_LAB_PATH}/scripts/utils/ensure_dependencies.py" "$1"
}

usage() {
    echo "Usage: ./go2_rl_lab.sh {--install|--list|--train|--play|--deploy-mujoco} [arguments]"
}

case "${1:-}" in
    -i|--install)
        "${python_exe}" -m pip install --no-deps -e "${go2_rl_lab_path}"
        ensure_optional_dependencies video
        ;;
    -l|--list)
        shift
        "${python_exe}" "${GO2_RL_LAB_PATH}/scripts/utils/list_envs.py" "$@"
        ;;
    -t|--train)
        shift
        "${python_exe}" "${GO2_RL_LAB_PATH}/scripts/rsl_rl/train.py" "$@"
        ;;
    -p|--play)
        shift
        "${python_exe}" "${GO2_RL_LAB_PATH}/scripts/rsl_rl/play.py" "$@"
        ;;
    --deploy-mujoco)
        shift
        ensure_optional_dependencies deploy
        "${python_exe}" "${GO2_RL_LAB_PATH}/deploy/deploy_mujoco/deploy_go2.py" "$@"
        ;;
    -h|--help|"")
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
