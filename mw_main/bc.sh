#!/bin/bash

envs=("Assembly" \
    "Basketball" \
    "CoffeePush" \
    "BoxClose" \
    "StickPull" \
    "StickPull" \
    "PegInsertSide" \
    "Soccer" \
    "button-press" \
    "pick-place" \
    "bin-picking" \
    "button-press-topdown" \
    "button-press-topdown-wall" \
    "door-lock" \
    "door-open" \
    "door-unlock" \
    "drawer-close" \
    "drawer-open" \
    "faucet-close" \
    "faucet-open" \
    "handle-press" \
    "handle-pull" \
    "handle-pull_side" \
    "lever-pull" \
    "window-close" \
    "window-open")


for env in ${envs[@]}; do
  python train_bc_mw.py --dataset.path ${env} --save_dir models/metaworld/${env} --use_wb 1 --wb_run ${env}
done
