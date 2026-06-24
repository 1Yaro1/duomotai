#!/usr/bin/env bash
set -o pipefail

cd /home/xuke/dyao/duomotai || exit 1

RUN_ID="${1:-tt_batch1_epoch5_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="/home/xuke/dyao/duomotai/result/label/TT/${RUN_ID}"
EXP_LOG="${RUN_DIR}/experiment.log"
MON_LOG="${RUN_DIR}/monitor.log"
SUMMARY_LOG="${RUN_DIR}/summary.log"
SCRIPT_SNAPSHOT="${RUN_DIR}/MindTS.sh.snapshot"
CONFIG_SNAPSHOT="${RUN_DIR}/unfixed_detect_label_multi_config.json.snapshot"
GIT_STATUS_SNAPSHOT="${RUN_DIR}/git_status.txt"
GIT_DIFF_SNAPSHOT="${RUN_DIR}/git_diff.patch"

mkdir -p "${RUN_DIR}"

cp scripts/multivariate_detection/detect_label/TT_script/MindTS.sh "${SCRIPT_SNAPSHOT}" 2>/dev/null || true
cp config/unfixed_detect_label_multi_config.json "${CONFIG_SNAPSHOT}" 2>/dev/null || true
git status --short > "${GIT_STATUS_SNAPSHOT}" 2>/dev/null || true
git diff -- scripts/multivariate_detection/detect_label/TT_script/MindTS.sh \
  ts_benchmark/baselines/MindTS/models/MindTS_model.py \
  ts_benchmark/baselines/utils.py \
  scripts/run_tt_with_monitor.sh \
  docs/tt_result_reproduction.md > "${GIT_DIFF_SNAPSHOT}" 2>/dev/null || true

{
  echo "RUN_ID=${RUN_ID}"
  echo "RUN_DIR=${RUN_DIR}"
  echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "SCRIPT=scripts/multivariate_detection/detect_label/TT_script/MindTS.sh"
  echo "SCRIPT_SNAPSHOT=${SCRIPT_SNAPSHOT}"
  echo "CONFIG_SNAPSHOT=${CONFIG_SNAPSHOT}"
  echo "GIT_STATUS_SNAPSHOT=${GIT_STATUS_SNAPSHOT}"
  echo "GIT_DIFF_SNAPSHOT=${GIT_DIFF_SNAPSHOT}"
  echo "CMD=PATH=/home/xuke/dyao/duomotai/.venv/bin:\$PATH PYTHONUNBUFFERED=1 bash scripts/multivariate_detection/detect_label/TT_script/MindTS.sh"
} | tee -a "${SUMMARY_LOG}"

(
  while true; do
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') ====="
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits || true
    echo "-- compute apps --"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
    echo "-- latest TT results --"
    ls -lt result/label/TT | head -n 12 || true
    echo
    sleep 30
  done
) >> "${MON_LOG}" 2>&1 &
MON_PID=$!
echo "MONITOR_PID=${MON_PID}" | tee -a "${SUMMARY_LOG}"

PATH="/home/xuke/dyao/duomotai/.venv/bin:${PATH}" PYTHONUNBUFFERED=1 \
  bash scripts/multivariate_detection/detect_label/TT_script/MindTS.sh 2>&1 | tee "${EXP_LOG}"
EXP_STATUS=${PIPESTATUS[0]}

{
  echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "EXIT_STATUS=${EXP_STATUS}"
  echo "-- final TT results --"
  ls -lt result/label/TT | head -n 20 || true
} | tee -a "${SUMMARY_LOG}"

kill "${MON_PID}" 2>/dev/null || true
wait "${MON_PID}" 2>/dev/null || true
exit "${EXP_STATUS}"
