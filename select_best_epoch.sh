#!/usr/bin/env bash
# Pick the best epoch by VALIDATION loss and confirm its checkpoint exists.
#
# train.py validates and saves inside the same `if epoch % save_epoch_freq == 0` block,
# so every epoch written to log.txt always has a matching <epoch>_net_G.pth. The test
# set is never consulted here.
#
# usage: ./select_best_epoch.sh <checkpoints_dir>/<experiment_name>
set -eo pipefail

EXP_DIR="${1:?usage: $0 <checkpoints_dir>/<experiment_name>}"
LOG="${EXP_DIR}/log.txt"

[[ -f "${LOG}" ]] || { echo "ABORT: no validation log at ${LOG}"; exit 1; }

echo "validation history (ranked by l1_avg_loss, lowest first):"
echo "    epoch   l1_avg_loss   mean_psnr   checkpoint"
# log lines look like:
#   Epoch  30   l1_avg_loss: 0.04213   mean_psnr: 22.145  std_psnr:1.802
BEST_EPOCH=""
while read -r ep l1 ps; do
  if [[ -f "${EXP_DIR}/${ep}_net_G.pth" ]]; then CK="present"; else CK="MISSING"; fi
  printf "  %7s   %11s   %9s   %s\n" "${ep}" "${l1}" "${ps}" "${CK}"
  if [[ -z "${BEST_EPOCH}" && "${CK}" == "present" ]]; then BEST_EPOCH="${ep}"; fi
done < <(grep -E '^Epoch' "${LOG}" \
         | sed -E 's/^Epoch +([0-9]+).*l1_avg_loss: *([0-9.eE+-]+).*mean_psnr: *([0-9.eE+-]+).*/\1 \2 \3/' \
         | sort -k2,2g)

if [[ -z "${BEST_EPOCH}" ]]; then
  echo "ABORT: no validated epoch has a saved checkpoint."
  exit 1
fi

echo
echo "BEST VALIDATION EPOCH: ${BEST_EPOCH}"
echo "checkpoint: ${EXP_DIR}/${BEST_EPOCH}_net_G.pth"
echo
echo "run the final test with:"
echo "  GPU_AUTHORIZED=1 GPU=<id> STAGE=test EPOCH=${BEST_EPOCH} ./run_resvit_3d.sh"
