#!/usr/bin/env bash
# 编排 TTA 补强(目标"把一部分做了"):
#  阶段1(已在跑): steps=3 全12-cell×3seed×7模式 -> tta_matrix_g{0,1}.csv
#  阶段2(本脚本等阶段1完后跑): steps=20 的 {entropy,physcalib,physent} 全12-cell
#     -> tta_s20_g{0,1}.csv,验证"Tent过度适应崩 / physcalib鲁棒 / physent能否稳住Tent"
# 完成后主叫方聚合两阶段出多基线+步数鲁棒性表。
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/tta

echo "[orch $(date '+%H:%M:%S')] 等 steps=3 主矩阵完成..."
i=0
while true; do
  g0=$(grep -c "WROTE" logs/tta/matrix_g0.log 2>/dev/null || echo 0)
  g1=$(grep -c "WROTE" logs/tta/matrix_g1.log 2>/dev/null || echo 0)
  if [ "$g0" -ge 1 ] && [ "$g1" -ge 1 ]; then echo "[orch] 阶段1完成"; break; fi
  if [ "$i" -ge 1200 ]; then echo "[orch] TIMEOUT 阶段1 (g0=$g0 g1=$g1)"; exit 1; fi
  i=$((i+1)); sleep 15
done

echo "[orch $(date '+%H:%M:%S')] 启动阶段2 steps=20"
rm -f result/tta_s20_g0.csv result/tta_s20_g1.csv
conda run -n ubench1 python tools/tta_matrix.py --sources busi,bus --seeds 41,42,43 \
  --modes naive,entropy,physcalib,physent --steps 20 --gpu 0 --out result/tta_s20_g0.csv \
  > logs/tta/s20_g0.log 2>&1 &
conda run -n ubench1 python tools/tta_matrix.py --sources BUSBRA --seeds 41,42,43 \
  --modes naive,entropy,physcalib,physent --steps 20 --gpu 1 --out result/tta_s20_g1.csv \
  > logs/tta/s20_g1.log 2>&1 &
wait
echo "[orch $(date '+%H:%M:%S')] 阶段2完成"
echo "s20_g0 rows=$(($(wc -l < result/tta_s20_g0.csv 2>/dev/null || echo 1)-1))  s20_g1 rows=$(($(wc -l < result/tta_s20_g1.csv 2>/dev/null || echo 1)-1))"
echo "=== TTA ORCH DONE ==="
