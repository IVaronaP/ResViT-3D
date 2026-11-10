#!/usr/bin/env bash
# Inference-only for ResViT 3D: run a chosen checkpoint over a chosen dataset.
#
# Same GPU guard and env preamble as run_resvit_3d.sh. Unlike the 2D test_resvit.sh,
# this writes into its OWN results directory keyed by patch size, so it can never mix
# its output into an older experiment's folder.
#
# usage: GPU_AUTHORIZED=1 GPU=<id> EPOCH=<n> ./test_resvit_3d.sh
set -eo pipefail

REPO_DIR="/home/export/personal/ivarpen/PhD/ResViT/ResViT-main-3d"
CONDA_ENV="${CONDA_ENV:-resvit_env_3d}"
BASE="/home/export/personal/ivarpen/PhD/ResViT"

GPU="${GPU:-1}"
MAX_USED_MIB="${MAX_USED_MIB:-2048}"
NEED_MIB="${NEED_MIB:-6000}"          # inference only, no optimizer state
PATCH="${PATCH:-128}"
PATCH_Z="${PATCH_Z:-48}"
OVERLAP="${OVERLAP:-32}"
EPOCH="${EPOCH:-latest}"
DATASET="${DATASET:-native_z}"
HOW_MANY="${HOW_MANY:-10000}"

TAG="z${PATCH_Z}_xy${PATCH}"
DATAROOT="${DATAROOT:-${BASE}/data_fibrillarin_3d/${DATASET}/}"
CKPT_DIR="${BASE}/checkpoints_fibrillarin_3d/${TAG}/"
RESULTS_DIR="${BASE}/results_fibrillarin_3d/${TAG}/"
EXP_FT="fibrillarin3d_BrightFluoro_resvit"

for _v in PATCH PATCH_Z; do
  _val="${!_v}"
  if (( _val <= 0 || _val % 16 != 0 )); then
    echo "ABORT: ${_v}=${_val} must be a POSITIVE MULTIPLE OF 16."
    echo "Each patch axis is reduced 16x before the transformer and restored afterwards;"
    echo "any other size breaks the ART skip-connection."
    exit 1
  fi
done

MIN_PATCH=$(( PATCH_Z < PATCH ? PATCH_Z : PATCH ))
if (( OVERLAP < 0 || OVERLAP >= MIN_PATCH )); then
  echo "ABORT: OVERLAP=${OVERLAP} must satisfy 0 <= OVERLAP < min(PATCH_Z,PATCH) = ${MIN_PATCH}."
  echo "An overlap equal to the patch collapses the sliding-window stride to 1 voxel."
  exit 1
fi

if [[ "${GPU_AUTHORIZED:-0}" != "1" ]]; then
  echo "ABORT: GPU use is not authorized."
  echo "Re-run with GPU_AUTHORIZED=1 once you have confirmed GPU ${GPU} is allocated to you."
  exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Using inherited CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (not overriding)."
  DEV="${CUDA_VISIBLE_DEVICES}"
else
  DEV="${GPU}"
fi
if [[ "${DEV}" == *,* ]]; then
  echo "ABORT: more than one device exposed (${DEV}). Single-GPU only."
  exit 1
fi
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
if ! nvidia-smi -i "${DEV}" >/dev/null 2>&1; then
  echo "ABORT: GPU index ${DEV} is not valid on this host."; exit 1
fi
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${DEV}")
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${DEV}")
if (( USED > MAX_USED_MIB )); then
  echo "ABORT: GPU ${DEV} already has ${USED} MiB in use (limit ${MAX_USED_MIB})."; exit 1
fi
if (( FREE < NEED_MIB )); then
  echo "ABORT: GPU ${DEV} has ${FREE} MiB free, need ${NEED_MIB} MiB."; exit 1
fi
export CUDA_VISIBLE_DEVICES="${DEV}"

CKPT="${CKPT_DIR}/${EXP_FT}/${EPOCH}_net_G.pth"
if [[ ! -f "${CKPT}" ]]; then
  echo "ABORT: no checkpoint for epoch '${EPOCH}': ${CKPT}"
  echo "Available epochs:"
  ls "${CKPT_DIR}/${EXP_FT}"/*_net_G.pth 2>/dev/null \
    | xargs -n1 basename 2>/dev/null | sed 's/_net_G.pth//' | sort -n || echo "  (none)"
  exit 1
fi

set +u
source ~/miniconda3/bin/activate "${CONDA_ENV}"
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${RESULTS_DIR}"
cd "${REPO_DIR}"

echo "==> inference with ${CKPT}"
python3 test.py \
  --dataroot "${DATAROOT}" \
  --name "${EXP_FT}" \
  --gpu_ids 0 \
  --model resvit_one \
  --which_model_netG resvit \
  --dataset_mode aligned \
  --norm batch \
  --phase test \
  --input_nc 1 \
  --output_nc 1 \
  --fineSize "${PATCH}" \
  --fineSizeZ "${PATCH_Z}" \
  --overlap "${OVERLAP}" \
  --nThreads 0 \
  --how_many "${HOW_MANY}" \
  --serial_batches \
  --display_id 0 \
  --results_dir "${RESULTS_DIR}" \
  --checkpoints_dir "${CKPT_DIR}" \
  --which_epoch "${EPOCH}"

echo "Results in: ${RESULTS_DIR}"
