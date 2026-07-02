#!/usr/bin/env bash
# Dump per-image val IoU for every (model, dataset, seed) needed by the Wilcoxon
# significance test. Round-robins jobs across GPU 0/1, 2 concurrent.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p logs/percase

# job = "MODEL DATASET EXP"
JOBS=(
  # PUMA-Net (USEANet) — busi=disc_mult02, others=c2a
  "USEANet busi disc_mult02_busi"
  "USEANet busi disc_mult02_s42"
  "USEANet busi disc_mult02_s43"
  "USEANet bus c2a_bus_s41"
  "USEANet bus c2a_bus_s42"
  "USEANet bus c2a_bus_s43"
  "USEANet BUSBRA c2a_BUSBRA_s41"
  "USEANet BUSBRA c2a_BUSBRA_s42"
  "USEANet BUSBRA c2a_BUSBRA_s43"
  "USEANet tuscui c2a_tuscui_s41"
  "USEANet tuscui c2a_tuscui_s42"
  "USEANet tuscui c2a_tuscui_s43"
  # H2Former — busi seed41=baseline_s41, else mtx_s*
  "H2Former busi baseline_s41"
  "H2Former busi mtx_s42"
  "H2Former busi mtx_s43"
  "H2Former bus mtx_s41"
  "H2Former bus mtx_s42"
  "H2Former bus mtx_s43"
  "H2Former BUSBRA mtx_s41"
  "H2Former BUSBRA mtx_s42"
  "H2Former BUSBRA mtx_s43"
  "H2Former tuscui mtx_s41"
  "H2Former tuscui mtx_s42"
  "H2Former tuscui mtx_s43"
  # MoE-off (USEANet nomoe) — busi/bus/BUSBRA only (tuscui has no nomoe)
  "USEANet busi nomoe_mult02_s41"
  "USEANet busi nomoe_mult02_s42"
  "USEANet busi nomoe_mult02_s43"
  "USEANet bus nomoe_bus_s41"
  "USEANet bus nomoe_bus_s42"
  "USEANet bus nomoe_bus_s43"
  "USEANet BUSBRA nomoe_BUSBRA_s41"
  "USEANet BUSBRA nomoe_BUSBRA_s42"
  "USEANet BUSBRA nomoe_BUSBRA_s43"
)

run_one() {
  local gpu="$1" model="$2" ds="$3" exp="$4"
  local out="result/percase_${model}_${ds}_${exp}.csv"
  if [[ -f "$out" ]]; then
    echo "[skip] $out exists"; return 0
  fi
  local log="logs/percase/${model}_${ds}_${exp}.log"
  conda run -n ubench1 python tools/dump_percase.py \
    --model "$model" --dataset_name "$ds" --exp_name "$exp" --gpu "$gpu" \
    > "$log" 2>&1
  grep "\[dump\]" "$log" | tail -1
}

i=0
for job in "${JOBS[@]}"; do
  read -r model ds exp <<< "$job"
  gpu=$(( i % 2 ))
  run_one "$gpu" "$model" "$ds" "$exp" &
  # launch in pairs (one per GPU), then wait
  if (( i % 2 == 1 )); then wait; fi
  i=$(( i + 1 ))
done
wait
echo "=== all dumps done ==="
ls -1 result/percase_*.csv | wc -l
