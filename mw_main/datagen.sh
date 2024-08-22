#!/bin/bash

envs=("Assembly" \
#     "Basketball" \
#     "CoffeePush" \
#     "BoxClose" \
#     "StickPull" \
#     "StickPull" \
#     "PegInsertSide" \
#     "Soccer" \
#     "button-press" \
#     "pick-place" \
#     "bin-picking" \
#     "button-press-topdown" \
#     "button-press-topdown-wall" \
#     "door-lock" \
#     "door-open" \
#     "door-unlock" \
    "drawer-close" \
    "drawer-open" \
    "faucet-close" \
    "faucet-open" \
    "handle-press" \
    "handle-pull" \
    "handle-pull-side" \
    "lever-pull" \
    "window-close" \
    "window-open")

for env in ${envs[@]}; do
  python generate_metaworld_dataset.py \
    --num_episodes 5 \
    --save_gifs 5 \
    --output_path data/metaworld/${env}_frame_stack_1_96x96_end_on_success \
    --env_cfg.env_name ${env} \
    --env_cfg.frame_stack 1 \
    --env_cfg.rl_image_size 96 \
    --env_cfg.end_on_success true
done