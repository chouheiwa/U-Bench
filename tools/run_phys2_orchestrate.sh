#!/usr/bin/env bash
# 编排: 等 matrix2 的 18 训练(busi/BUSBRA × nak/att/snr × 3seed)全部完成,
# 再自动跑 busi(GPU0) + BUSBRA(GPU1) 的逐物理量跨域评测(各含 full=calib_源),
# 落 per-case 到 result/percase_cross_dataset.csv 供 wilcoxon_cross.py 按源分组。
# 长任务(数小时);完成后主叫方跑 tools/wilcoxon_cross.py 出缩放证据。
# 用法: bash tools/run_phys2_orchestrate.sh
set -u
cd /home/chouheiwa/python/U-Bench

EXPS=""
for src in busi BUSBRA; do for v in nak att snr; do for sd in 41 42 43; do
  EXPS="$EXPS ./output/USEANet/${src}/calib${v}_${src}_s${sd}/training.log"
done; done; done

count_done() {
  local n=0
  for lg in $EXPS; do
    grep -q "Training completed" "$lg" 2>/dev/null && n=$((n+1))
  done
  echo "$n"
}

echo "[orch $(date '+%H:%M:%S')] 等 18 训练完成..."
# 最长等 10h(3600 次 × 10s),每 2min 打点
i=0
while true; do
  d=$(count_done)
  if [ "$d" -ge 18 ]; then echo "[orch $(date '+%H:%M:%S')] 18/18 训练完成"; break; fi
  if [ "$i" -ge 3600 ]; then echo "[orch $(date '+%H:%M:%S')] TIMEOUT: 仅 $d/18 完成, 中止"; exit 1; fi
  [ $((i % 12)) -eq 0 ] && echo "[orch $(date '+%H:%M:%S')] 进度 $d/18"
  i=$((i+1)); sleep 10
done

echo "[orch $(date '+%H:%M:%S')] 启动 busi(GPU0) + BUSBRA(GPU1) 跨域评测"
bash tools/run_phys_cross_matrix.sh 0 busi   > logs/physabl/cross_busi.log   2>&1 &
bash tools/run_phys_cross_matrix.sh 1 BUSBRA > logs/physabl/cross_BUSBRA.log 2>&1 &
wait
echo "[orch $(date '+%H:%M:%S')] 跨域评测完成"
echo "busi   n_xds=$(grep -c '^\[xds\]' logs/physabl/cross_busi.log)   n_err=$(grep -cE 'size mismatch|Unexpected|Missing key|Traceback' logs/physabl/cross_busi.log)"
echo "BUSBRA n_xds=$(grep -c '^\[xds\]' logs/physabl/cross_BUSBRA.log) n_err=$(grep -cE 'size mismatch|Unexpected|Missing key|Traceback' logs/physabl/cross_BUSBRA.log)"
echo "=== ORCH DONE ==="
