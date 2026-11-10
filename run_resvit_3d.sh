#!/usr/bin/env bash
# ResViT 3D driver. Mirrors run_resvit.sh (which is left untouched), with:
#   - STAGE required, so nothing runs unless you ask for it explicitly
#   - GPU_AUTHORIZED required: a free GPU is NOT an authorized GPU on a shared server
#   - checkpoints/results keyed by patch size, so experiments never overwrite each other
set -eo pipefail

REPO_DIR="/home/export/personal/ivarpen/PhD/ResViT/ResViT-main-3d"
CONDA_ENV="${CONDA_ENV:-resvit_env_3d}"       # resvit_env (2D) is never touched
BASE="/home/export/personal/ivarpen/PhD/ResViT"

###################################
# CONFIG (all overridable from the environment)
###################################
GPU="${GPU:-1}"                 # physical card, matching the 2D convention
MAX_USED_MIB="${MAX_USED_MIB:-2048}"
NEED_MIB="${NEED_MIB:-9000}"
PATCH="${PATCH:-128}"           # XY patch, must be divisible by 16
PATCH_Z="${PATCH_Z:-48}"        # Z patch, must be divisible by 16 (48 => no Z padding)
BATCH="${BATCH:-1}"
PPV="${PPV:-8}"                 # train patches per volume per epoch
VPPV="${VPPV:-4}"               # fixed val patches per volume
OVERLAP="${OVERLAP:-32}"
EPOCH="${EPOCH:-latest}"        # checkpoint the test stage loads
DATASET="${DATASET:-native_z}"
STAGE="${STAGE:?set STAGE=pre|ft|test explicitly (there is no default)}"

TAG="z${PATCH_Z}_xy${PATCH}"
DATAROOT="${BASE}/data_fibrillarin_3d/${DATASET}/"
CKPT_DIR="${BASE}/checkpoints_fibrillarin_3d/${TAG}/"
RESULTS_DIR="${BASE}/results_fibrillarin_3d/${TAG}/"
EXP_PRE="fibrillarin3d_BrightFluoro_pre_trained"
EXP_FT="fibrillarin3d_BrightFluoro_resvit"

INPUT_NC=1
OUTPUT_NC=1
LAMBDA_A=100
LR_PRE=0.0002
LR_FT=0.001
NITER_PRE=50
NDECAY_PRE=50
NITER_FT=25
NDECAY_FT=25
SAVE_FREQ=5

###################################
# CONFIG VALIDATION (before any GPU or model work)
###################################
for _v in PATCH PATCH_Z; do
  _val="${!_v}"
  if (( _val <= 0 || _val % 16 != 0 )); then
    echo "ABORT: ${_v}=${_val} must be a POSITIVE MULTIPLE OF 16."
    echo "Each patch axis is reduced 16x before the transformer (encoder /4, ART /4) and"
    echo "restored afterwards; any other size breaks the ART skip-connection, which is the"
    echo "same failure mode as fineSize=624 in the 2D pipeline."
    exit 1
  fi
done

MIN_PATCH=$(( PATCH_Z < PATCH ? PATCH_Z : PATCH ))
if (( OVERLAP < 0 || OVERLAP >= MIN_PATCH )); then
  echo "ABORT: OVERLAP=${OVERLAP} must satisfy 0 <= OVERLAP < min(PATCH_Z,PATCH) = ${MIN_PATCH}."
  echo "An overlap equal to the patch collapses the sliding-window stride to 1 voxel,"
  echo "which inflates the window count by orders of magnitude."
  exit 1
fi

###################################
# SHARED-SERVER GPU GUARD
###################################
if [[ "${GPU_AUTHORIZED:-0}" != "1" ]]; then
  echo "ABORT: GPU use is not authorized."
  echo "Re-run with GPU_AUTHORIZED=1 once you have confirmed GPU ${GPU} is allocated to you."
  echo "An idle GPU is not an authorized GPU on a shared server."
  exit 1
fi

# Honour an inherited allocation, but still validate it.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Using inherited CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (not overriding)."
  DEV="${CUDA_VISIBLE_DEVICES}"
else
  DEV="${GPU}"
fi

if [[ "${DEV}" == *,* ]]; then
  echo "ABORT: more than one device exposed (${DEV}). This workflow is single-GPU only."
  exit 1
fi

nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
if ! nvidia-smi -i "${DEV}" >/dev/null 2>&1; then
  echo "ABORT: GPU index ${DEV} is not valid on this host."
  exit 1
fi
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${DEV}")
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${DEV}")
if (( USED > MAX_USED_MIB )); then
  echo "ABORT: GPU ${DEV} already has ${USED} MiB in use (limit ${MAX_USED_MIB})."
  echo "Another researcher may be running there. Choose a card explicitly: GPU=<id> STAGE=${STAGE} $0"
  exit 1
fi
if (( FREE < NEED_MIB )); then
  echo "ABORT: GPU ${DEV} has ${FREE} MiB free but ${NEED_MIB} MiB is needed for ${PATCH_Z}x${PATCH}x${PATCH}."
  echo "Reduce the patch (e.g. PATCH_Z=32) or pick another authorized card."
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${DEV}"
echo "GPU ${DEV}: ${FREE} MiB free, ${USED} MiB in use."

###################################
# ENV
###################################
set +u
source ~/miniconda3/bin/activate "${CONDA_ENV}"
set -u
# same thread pinning as the 2D script ("para evitar el stack smashing")
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# unbuffered stdout, so a nohup log updates live instead of in 8 KB chunks
export PYTHONUNBUFFERED=1
# without this CUDA is not found
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${CKPT_DIR}" "${RESULTS_DIR}"
cd "${REPO_DIR}"        # the pretrained ViT path in transformer_configs.py is relative

COMMON="--dataroot ${DATAROOT} --gpu_ids 0 --model resvit_one --dataset_mode aligned \
  --which_direction AtoB --norm batch --pool_size 0 \
  --input_nc ${INPUT_NC} --output_nc ${OUTPUT_NC} \
  --fineSize ${PATCH} --fineSizeZ ${PATCH_Z} \
  --patches_per_volume ${PPV} --val_patches_per_volume ${VPPV} \
  --batchSize ${BATCH} --nThreads 0 --display_id 0 --checkpoints_dir ${CKPT_DIR}"

###################################
# STAGES
###################################
case "${STAGE}" in
  pre)
    echo "==> STAGE 1/3  pre-train (res_cnn) -> ${EXP_PRE}   patch ${PATCH_Z}x${PATCH}x${PATCH}"
    python3 train.py ${COMMON} \
      --name "${EXP_PRE}" \
      --which_model_netG res_cnn \
      --lambda_A "${LAMBDA_A}" \
      --lr "${LR_PRE}" \
      --niter "${NITER_PRE}" \
      --niter_decay "${NDECAY_PRE}" \
      --save_epoch_freq "${SAVE_FREQ}"
    ;;

  ft)
    PRE="${CKPT_DIR}/${EXP_PRE}/latest_net_G.pth"
    if [[ ! -f "${PRE}" ]]; then
      echo "ABORT: stage-1 checkpoint missing: ${PRE}"
      echo "Run STAGE=pre first. A 2D checkpoint cannot be used here: every conv gained a Z axis."
      exit 1
    fi
    echo "==> STAGE 2/3  fine-tune (resvit) -> ${EXP_FT}, warm start from ${PRE}"
    python3 train.py ${COMMON} \
      --name "${EXP_FT}" \
      --which_model_netG resvit \
      --lambda_A "${LAMBDA_A}" \
      --lr "${LR_FT}" \
      --niter "${NITER_FT}" \
      --niter_decay "${NDECAY_FT}" \
      --save_epoch_freq "${SAVE_FREQ}" \
      --pre_trained_transformer 1 \
      --pre_trained_resnet 1 \
      --pre_trained_path "${PRE}"
    ;;

  test)
    CKPT="${CKPT_DIR}/${EXP_FT}/${EPOCH}_net_G.pth"
    if [[ ! -f "${CKPT}" ]]; then
      echo "ABORT: no checkpoint for epoch '${EPOCH}': ${CKPT}"
      echo "Available epochs:"
      ls "${CKPT_DIR}/${EXP_FT}"/*_net_G.pth 2>/dev/null \
        | xargs -n1 basename 2>/dev/null | sed 's/_net_G.pth//' | sort -n || echo "  (none)"
      echo "Pick the best VALIDATION epoch with: ./select_best_epoch.sh ${CKPT_DIR}/${EXP_FT}"
      exit 1
    fi
    echo "==> STAGE 3/3  test using ${CKPT}"
    python3 test.py ${COMMON} \
      --name "${EXP_FT}" \
      --which_model_netG resvit \
      --phase test \
      --serial_batches \
      --how_many 10000 \
      --overlap "${OVERLAP}" \
      --results_dir "${RESULTS_DIR}" \
      --which_epoch "${EPOCH}"
    ;;

  *)
    echo "STAGE must be pre|ft|test (got '${STAGE}')"
    exit 1
    ;;
esac

echo "Done. checkpoints: ${CKPT_DIR}   results: ${RESULTS_DIR}"
