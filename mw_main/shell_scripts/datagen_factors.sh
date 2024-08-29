#!/bin/bash

envs=(  #"Assembly" \
     #"Basketball" \
    # "CoffeePush" \
    # "BoxClose" \
    # "StickPull" \
    # "PegInsertSide" \
    # "Soccer" \
    "button-press" \
    "pick-place" \
    #"bin-picking" \
    # "button-press-topdown" \
    # "button-press-topdown-wall" \
    "door-lock" \
    "door-open" \
    # "door-unlock" \
    "drawer-close" \
    # "drawer-open" \
    #"faucet-close" \
    "faucet-open" \
    "handle-press" \
    "handle-pull" \
    # "handle-pull-side" \
    "lever-pull" \
    # "window-close" \
    "window-open" \
    )

factors=(
    # "arm_pos" \
    # "camera_pos" \
    # # "distractor_pos"  \
    # "floor_texture" \
    # #"object_texture" \
    # "table_pos" \
    # "table_texture" \
    "light" \
    #"object_size"
)

max_processes=10
n=0

for env in ${envs[@]}; do
    for factor in ${factors[@]}; do
        python generate_metaworld_dataset.py \
            --num_episodes 10 \
            --save_gifs 5 \
            --output_path bc_data/metaworld/${env}_${factor}_frame_stack_1_96x96_end_on_success \
            --env_cfg.env_name ${env} \
            --env_cfg.factor_kwargs [${factor}] \
            --env_cfg.frame_stack 1 \
            --env_cfg.rl_image_size 96 \
            --env_cfg.end_on_success true &
        n=$((n + 1))
        if test $n -eq $max_processes; then
            wait -n 
            n=0
        fi
    done
done
