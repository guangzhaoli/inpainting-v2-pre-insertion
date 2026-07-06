#!/usr/bin/env bash
set -euo pipefail

# Submit a DLC PyTorchJob for the inpainting-v2-pre-insertion project.
#
# The remote job runs this repo's run_multi_node_dlc.sh, which auto-selects
# configs/insert_anything_${WORKERS * WORKER_GPU}.yaml plus the matching
# accelerate_config_${WORKERS * WORKER_GPU}.yaml unless CONFIG_FILE or
# ACCELERATE_CONFIG is set.
# Override the config with: CONFIG_FILE=configs/insert_anything_teacher_human.yaml bash dlc.sh
# Multi-worker example: WORKERS=4 WORKER_GPU=8 CONFIG_FILE=configs/insert_anything_teacher_human.yaml bash dlc.sh
# Dry-run (print the resolved `dlc submit ...` command without submitting): bash dlc.sh --dry-run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

DLC_BIN="${DLC_BIN:-/mnt/cpfs/weiqingyan/dlc}"
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-/mnt/cpfs/weiqingyan/miniconda3/envs/qwen-inpainting-pre-insertion}"
CONDA_ACTIVATE_SH="${CONDA_ACTIVATE_SH:-/mnt/cpfs/weiqingyan/miniconda3/etc/profile.d/conda.sh}"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-run_multi_node_dlc.sh}"

PRIORITY="${PRIORITY:-6}"
WORKSPACE_ID="${WORKSPACE_ID:-240810}"
RESOURCE_ID="${RESOURCE_ID:-quotaev2tl4w6aw0}"
WORKERS="${WORKERS:-1}"
WORKER_GPU="${WORKER_GPU:-8}"
WORKER_CPU="${WORKER_CPU:-116}"
WORKER_MEMORY="${WORKER_MEMORY:-1800Gi}"
WORKER_SHARED_MEMORY="${WORKER_SHARED_MEMORY:-1800Gi}"
WORKER_IMAGE="${WORKER_IMAGE:-pai-wl-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/acr_namespace/weiqingyan:devplus_accelerated}"
JOB_MAX_RUNNING_TIME_MINUTES="${JOB_MAX_RUNNING_TIME_MINUTES:-2880}"

MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-19001}"
NUM_PROCESSES="${NUM_PROCESSES:-$((WORKERS * WORKER_GPU))}"

DATA_SOURCE_URIS="${DATA_SOURCE_URIS:-cpfs://cpfs-298fffb575a502fe.cn-wulanchabu/ptc-29f47d9393ad2b16/exp-29f2869e7d984aa6/::/mnt/cpfs,nas://292a8d49e93-kgi71.cn-wulanchabu.nas.aliyuncs.com/::/mnt/nasB,oss://pai-wlcb-ai-oss.oss-cn-wulanchabu-internal.aliyuncs.com/::/mnt/oss}"
VPC_ID="${VPC_ID:-vpc-0jl5rpw5qokp6p2ettip6}"
SWITCH_ID="${SWITCH_ID:-vsw-0jlmr9rjzed093yr9c0kz}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-sg-0jl0pd5qaerdj75wmred}"
EXTENDED_CIDRS="${EXTENDED_CIDRS:-10.1.255.0/29,10.1.255.8/29,10.1.16.0/20}"

HF_HOME="${HF_HOME:-/mnt/cpfs/yangyicun/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
CONFIG_FILE="${CONFIG_FILE:-}"
RESUME="${RESUME:-}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-}"
REQUIRE_MASTER_ADDR="${REQUIRE_MASTER_ADDR:-0}"

# Forwarded to the remote script as extra args (after `--`).
REMOTE_EXTRA_ARGS=()
DRY_RUN=0

parse_submit_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      --)
        shift
        REMOTE_EXTRA_ARGS+=("$@")
        break
        ;;
      *)
        echo "Unknown argument: $1" >&2
        echo "Use --dry-run for submit dry-run, or pass remote script args after --." >&2
        exit 2
        ;;
    esac
  done
}

validate_remote_args() {
  if (( ${#REMOTE_EXTRA_ARGS[@]} > 1 )); then
    echo "Only one remote argument is supported, and it must be the config path." >&2
    echo "Use CONFIG_FILE=... or pass exactly one config after --." >&2
    exit 2
  fi
  if (( ${#REMOTE_EXTRA_ARGS[@]} == 1 )) && [[ -n "${CONFIG_FILE}" ]]; then
    echo "Specify the training config either with CONFIG_FILE=... or after --, not both." >&2
    exit 2
  fi
}

shell_join() {
  local out="" q arg
  for arg in "$@"; do
    printf -v q '%q' "${arg}"
    out+="${q} "
  done
  printf '%s' "${out% }"
}

shell_quote() {
  local q
  printf -v q '%q' "$1"
  printf '%s' "${q}"
}

join_remote_steps() {
  local out="" step
  for step in "$@"; do
    if [[ -n "${out}" ]]; then
      out+="; "
    fi
    out+="${step}"
  done
  printf '%s' "${out}"
}

project_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s' "${path}"
  else
    printf '%s/%s' "${PROJECT_DIR}" "${path}"
  fi
}

parse_job_id() {
  local log_file="$1"
  local id=""
  id="$(
    grep -Eio 'job[_ -]?id[^[:alnum:]]+dlc[-a-z0-9]+' "${log_file}" \
      | tail -n 1 \
      | grep -Eio 'dlc[-a-z0-9]+$' \
      || true
  )"
  printf '%s' "${id}"
}

state_assign() {
  local key="$1"
  local value="$2"
  printf '%s=' "${key}"
  printf '%q' "${value}"
  printf '\n'
}

priority_guard() {
  if ! [[ "${PRIORITY}" =~ ^[0-9]+$ ]]; then
    echo "PRIORITY must be an integer, got: ${PRIORITY}" >&2
    exit 2
  fi
  if (( PRIORITY > 6 )); then
    echo "Refusing to submit DLC job with priority=${PRIORITY}; maximum allowed is 6." >&2
    exit 6
  fi
}

resource_guard() {
  if ! [[ "${WORKERS}" =~ ^[0-9]+$ ]] || (( WORKERS < 1 )); then
    echo "WORKERS must be a positive integer, got: ${WORKERS}" >&2
    exit 2
  fi
  if ! [[ "${WORKER_GPU}" =~ ^[0-9]+$ ]] || (( WORKER_GPU < 1 )); then
    echo "WORKER_GPU must be a positive integer, got: ${WORKER_GPU}" >&2
    exit 2
  fi
  if ! [[ "${NUM_PROCESSES}" =~ ^[0-9]+$ ]] || (( NUM_PROCESSES < 1 )); then
    echo "NUM_PROCESSES must be a positive integer, got: ${NUM_PROCESSES}" >&2
    exit 2
  fi
  if (( NUM_PROCESSES != WORKERS * WORKER_GPU )); then
    echo "NUM_PROCESSES should equal WORKERS * WORKER_GPU for accelerate." >&2
    echo "Got NUM_PROCESSES=${NUM_PROCESSES}, WORKERS=${WORKERS}, WORKER_GPU=${WORKER_GPU}." >&2
    exit 2
  fi
  if (( WORKER_CPU > 124 )); then
    echo "WORKER_CPU must not exceed the known per-node limit 124, got: ${WORKER_CPU}" >&2
    exit 2
  fi
}

parse_submit_args "$@"
validate_remote_args
priority_guard
resource_guard

if [[ -z "${MASTER_ADDR}" && "${WORKERS}" -eq 1 ]]; then
  MASTER_ADDR="127.0.0.1"
fi
if [[ -z "${MASTER_ADDR}" && "${WORKERS}" -gt 1 && "${REQUIRE_MASTER_ADDR}" == "1" ]]; then
  echo "MASTER_ADDR is required for multi-worker submission when REQUIRE_MASTER_ADDR=1." >&2
  echo "Unset REQUIRE_MASTER_ADDR to rely on DLC-injected MASTER_ADDR." >&2
  exit 2
fi

default_config_path="configs/insert_anything_${NUM_PROCESSES}.yaml"
remote_config_arg=""
if (( ${#REMOTE_EXTRA_ARGS[@]} == 1 )); then
  remote_config_arg="${REMOTE_EXTRA_ARGS[0]}"
fi
effective_config_path="${CONFIG_FILE:-${remote_config_arg:-${default_config_path}}}"
effective_accelerate_config="${ACCELERATE_CONFIG:-accelerate_config_${NUM_PROCESSES}.yaml}"

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ ! -x "${DLC_BIN}" ]]; then
    echo "Missing executable DLC_BIN: ${DLC_BIN}" >&2
    exit 5
  fi
  if [[ ! -d "${PROJECT_DIR}" || ! -f "${PROJECT_DIR}/run_multi_node_dlc.sh" ]]; then
    echo "Missing inpainting project dir or run_multi_node_dlc.sh: ${PROJECT_DIR}" >&2
    exit 5
  fi
  if [[ ! -f "${PROJECT_DIR}/${REMOTE_SCRIPT}" ]]; then
    echo "Missing remote script: ${PROJECT_DIR}/${REMOTE_SCRIPT}" >&2
    exit 5
  fi
  if [[ ! -f "${CONDA_ACTIVATE_SH}" ]]; then
    echo "Missing conda activation script: ${CONDA_ACTIVATE_SH}" >&2
    exit 5
  fi
  if [[ ! -f "$(project_path "${effective_config_path}")" ]]; then
    echo "Missing CONFIG_FILE: ${effective_config_path}" >&2
    exit 5
  fi
  if [[ ! -f "$(project_path "${effective_accelerate_config}")" ]]; then
    echo "Missing ACCELERATE_CONFIG: ${effective_accelerate_config}" >&2
    echo "Set ACCELERATE_CONFIG explicitly, create accelerate_config_${NUM_PROCESSES}.yaml, or use an existing process count such as 4, 8, or 32." >&2
    exit 5
  fi
fi

run_tag="$(date +%Y%m%d_%H%M%S)"
job_kind="inpainting_train"
job_name="${JOB_NAME:-inpainting-${job_kind}-${run_tag}}"
remote_call="$(shell_join bash "${REMOTE_SCRIPT}" "${REMOTE_EXTRA_ARGS[@]}")"

remote_steps=(
  "set -eo pipefail"
  "cd $(shell_quote "${PROJECT_DIR}")"
  ". $(shell_quote "${CONDA_ACTIVATE_SH}")"
  "conda activate $(shell_quote "${REMOTE_CONDA_ENV}")"
  'export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"'
  "export PROJECT_DIR=$(shell_quote "${PROJECT_DIR}")"
  "export MASTER_PORT=$(shell_quote "${MASTER_PORT}")"
  "export WORLD_SIZE=$(shell_quote "${WORKERS}")"
  "export NPROC_PER_NODE=$(shell_quote "${WORKER_GPU}")"
  "export NUM_PROCESSES=$(shell_quote "${NUM_PROCESSES}")"
  "export TOTAL_PROCESSES=$(shell_quote "${NUM_PROCESSES}")"
  "export MIXED_PRECISION=$(shell_quote "${MIXED_PRECISION}")"
  "export PYTHONPATH=$(shell_quote "${PROJECT_DIR}"):\${PYTHONPATH:-}"
  "export HF_HOME=$(shell_quote "${HF_HOME}")"
  "export HF_DATASETS_CACHE=$(shell_quote "${HF_DATASETS_CACHE}")"
  "export HF_DATASETS_OFFLINE=$(shell_quote "${HF_DATASETS_OFFLINE}")"
  "export TRANSFORMERS_OFFLINE=$(shell_quote "${TRANSFORMERS_OFFLINE}")"
  "export NCCL_DEBUG=WARN"
  "export OMP_NUM_THREADS=1"
  "export MKL_NUM_THREADS=1"
  "export OPENBLAS_NUM_THREADS=1"
  "export NUMEXPR_NUM_THREADS=1"
  "export VECLIB_MAXIMUM_THREADS=1"
  "export BLIS_NUM_THREADS=1"
)

if [[ -n "${MASTER_ADDR}" ]]; then
  remote_steps+=("export MASTER_ADDR=$(shell_quote "${MASTER_ADDR}")")
fi
if [[ -n "${CONFIG_FILE}" ]]; then
  remote_steps+=("export CONFIG_FILE=$(shell_quote "${CONFIG_FILE}")")
fi
if [[ -n "${RESUME}" ]]; then
  remote_steps+=("export RESUME=$(shell_quote "${RESUME}")")
fi
if [[ -n "${ACCELERATE_CONFIG}" ]]; then
  remote_steps+=("export ACCELERATE_CONFIG=$(shell_quote "${ACCELERATE_CONFIG}")")
fi

remote_steps+=("${remote_call}")
remote_command_body="$(join_remote_steps "${remote_steps[@]}")"
run_command="/bin/bash -c $(shell_quote "${remote_command_body}")"

if [[ -n "${CONFIG_FILE}" ]]; then
  config_display="${CONFIG_FILE}"
elif [[ -n "${remote_config_arg}" ]]; then
  config_display="<arg: ${remote_config_arg}>"
else
  config_display="<auto: ${default_config_path}>"
fi

cmd=(
  "${DLC_BIN}" submit pytorchjob
  --name="${job_name}"
  --command="${run_command}"
  --resource_id="${RESOURCE_ID}"
  --workspace_id="${WORKSPACE_ID}"
  --vpc_id="${VPC_ID}"
  --switch_id="${SWITCH_ID}"
  --security_group_id="${SECURITY_GROUP_ID}"
  --extended_cidrs="${EXTENDED_CIDRS}"
  --data_source_uris="${DATA_SOURCE_URIS}"
  --worker_image="${WORKER_IMAGE}"
  --worker_cpu="${WORKER_CPU}"
  --worker_memory="${WORKER_MEMORY}"
  --worker_shared_memory="${WORKER_SHARED_MEMORY}"
  --worker_gpu="${WORKER_GPU}"
  --priority="${PRIORITY}"
  --job_max_running_time_minutes="${JOB_MAX_RUNNING_TIME_MINUTES}"
  --workers="${WORKERS}"
)

echo "Submitting inpainting DLC job: ${job_name}"
echo "Kind: ${job_kind}"
echo "Remote script: ${REMOTE_SCRIPT} ${REMOTE_EXTRA_ARGS[*]:-}"
echo "Project dir: ${PROJECT_DIR}"
echo "Remote conda env: ${REMOTE_CONDA_ENV}"
echo "Workers: ${WORKERS} x ${WORKER_GPU} GPU"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "CONFIG_FILE: ${config_display}"
echo "MIXED_PRECISION: ${MIXED_PRECISION}"
echo "RESUME: ${RESUME:-<none>}"
echo "MASTER_ADDR: ${MASTER_ADDR:-<DLC injected>}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "Priority: ${PRIORITY}"
echo "Workspace: ${WORKSPACE_ID}"
echo "Resource: ${RESOURCE_ID}"
echo "HF_HOME: ${HF_HOME}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[DRY-RUN] '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${PROJECT_DIR}/dlc_jobs/logs" "${PROJECT_DIR}/dlc_jobs/state"
submit_log="${PROJECT_DIR}/dlc_jobs/logs/submit_${job_kind}_${run_tag}.log"
set +e
"${cmd[@]}" 2>&1 | tee "${submit_log}"
submit_rc=${PIPESTATUS[0]}
set -e
if [[ "${submit_rc}" != "0" ]]; then
  echo "DLC submit failed; see ${submit_log}" >&2
  exit "${submit_rc}"
fi

job_id="$(parse_job_id "${submit_log}")"
if [[ -z "${job_id}" ]]; then
  echo "Could not parse JobId from ${submit_log}" >&2
  exit 4
fi

state_file="${PROJECT_DIR}/dlc_jobs/state/${job_kind}.env"
remote_extra_args_quoted="$(shell_join "${REMOTE_EXTRA_ARGS[@]}")"
{
  state_assign "JOB_ID" "${job_id}"
  state_assign "JOB_NAME" "${job_name}"
  state_assign "JOB_KIND" "${job_kind}"
  state_assign "REMOTE_SCRIPT" "${REMOTE_SCRIPT}"
  printf 'REMOTE_EXTRA_ARGS=%s\n' "${remote_extra_args_quoted}"
  state_assign "PROJECT_DIR" "${PROJECT_DIR}"
  state_assign "REMOTE_CONDA_ENV" "${REMOTE_CONDA_ENV}"
  state_assign "PRIORITY" "${PRIORITY}"
  state_assign "WORKERS" "${WORKERS}"
  state_assign "WORKER_GPU" "${WORKER_GPU}"
  state_assign "NUM_PROCESSES" "${NUM_PROCESSES}"
  state_assign "WORKER_CPU" "${WORKER_CPU}"
  state_assign "WORKER_MEMORY" "${WORKER_MEMORY}"
  state_assign "WORKER_SHARED_MEMORY" "${WORKER_SHARED_MEMORY}"
  state_assign "WORKER_IMAGE" "${WORKER_IMAGE}"
  state_assign "MASTER_ADDR" "${MASTER_ADDR}"
  state_assign "MASTER_PORT" "${MASTER_PORT}"
  state_assign "CONFIG_FILE" "${CONFIG_FILE}"
  state_assign "EFFECTIVE_CONFIG_FILE" "${effective_config_path}"
  state_assign "ACCELERATE_CONFIG" "${ACCELERATE_CONFIG}"
  state_assign "EFFECTIVE_ACCELERATE_CONFIG" "${effective_accelerate_config}"
  state_assign "RESUME" "${RESUME}"
  state_assign "MIXED_PRECISION" "${MIXED_PRECISION}"
  state_assign "HF_HOME" "${HF_HOME}"
  state_assign "HF_DATASETS_CACHE" "${HF_DATASETS_CACHE}"
  state_assign "SUBMIT_LOG" "${submit_log}"
  state_assign "SUBMITTED_AT" "$(date -Is)"
} > "${state_file}"
cp "${state_file}" "${PROJECT_DIR}/dlc_jobs/state/latest.env"

echo "DLC inpainting job submitted: ${job_name}"
echo "JobId: ${job_id}"
echo "State: ${state_file}"
