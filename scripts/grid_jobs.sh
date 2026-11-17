#!/bin/bash
# Main pilot grid.  Usage: scripts/grid_jobs.sh MUON_LR ADAMW_LR SGD_LR [SGD_EXTRA]
MU=$1; AD=$2; SG=$3; SGX=${4:-}; AUX=${AUX:-1e-3}  # Muon aux-AdamW lr (as swept)
C="--y_mode sep --readin_init gauss --save_model --eval_every 100 --qshift_every 500"
for cond in iso 1/i-ali 1/i2-ali 1/i2-rot; do
  t=$(echo $cond | tr -d '/')
  echo "--opt muon --lr $MU --aux_lr $AUX --cond $cond --steps 6000 $C --out runs/grid/muon_${t}_s0.json"
done
for cond in iso 1/i-ali 1/i2-ali 1/i2-rot; do
  t=$(echo $cond | tr -d '/')
  echo "--opt adamw --lr $AD --cond $cond --steps 10000 $C --out runs/grid/adamw_${t}_s0.json"
done
for cond in 1/i2-ali 1/i2-rot; do
  t=$(echo $cond | tr -d '/')
  echo "--opt muon --lr $MU --aux_lr $AUX --muon_params hidden+in --cond $cond --steps 6000 $C --out runs/grid/muonin_${t}_s0.json"
done
for cond in iso 1/i2-ali; do
  t=$(echo $cond | tr -d '/')
  echo "--opt sgd --lr $SG $SGX --cond $cond --steps 10000 $C --out runs/grid/sgd_${t}_s0.json"
done
for cond in iso 1/i2-ali; do
  t=$(echo $cond | tr -d '/')
  echo "--opt muon --lr $MU --aux_lr $AUX --cond $cond --steps 6000 --seed 1 $C --out runs/grid/muon_${t}_s1.json"
  echo "--opt adamw --lr $AD --cond $cond --steps 10000 --seed 1 $C --out runs/grid/adamw_${t}_s1.json"
done
