#!/usr/bin/env bash
# Fill the 5 missing steps=20 cross cells so the collapse-prevention table (§4)
# covers all 9 cross cells x 3 seeds. modes: naive,entropy,physcalib
# (physent == entropy already established, skipped to save compute).
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/gate
M="naive,entropy,physcalib"

# GPU0: busi->BrEaST  and  bus->{busi,BUSBRA}
(
  conda run -n ubench1 python tools/tta_matrix.py --sources busi --targets BrEaST \
    --seeds 41,42,43 --steps 20 --lr 1e-3 --gpu 0 --modes "$M" --out result/tta_s20_c.csv
  conda run -n ubench1 python tools/tta_matrix.py --sources bus --targets busi,BUSBRA \
    --seeds 41,42,43 --steps 20 --lr 1e-3 --gpu 0 --modes "$M" --out result/tta_s20_c.csv
) > logs/gate/s20_fill_gpu0.log 2>&1 &
P0=$!

# GPU1: BUSBRA->{bus,BrEaST}
(
  conda run -n ubench1 python tools/tta_matrix.py --sources BUSBRA --targets bus,BrEaST \
    --seeds 41,42,43 --steps 20 --lr 1e-3 --gpu 1 --modes "$M" --out result/tta_s20_d.csv
) > logs/gate/s20_fill_gpu1.log 2>&1 &
P1=$!

wait $P0 $P1
echo "ALL_DONE c=$(wc -l < result/tta_s20_c.csv 2>/dev/null) d=$(wc -l < result/tta_s20_d.csv 2>/dev/null)"
