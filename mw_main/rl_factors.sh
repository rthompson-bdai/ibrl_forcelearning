#!/bin/bash


envs=(  #"Assembly" \
     #"Basketball" \
    # "CoffeePush" \
    # "BoxClose" \
    # "StickPull" \
    # "PegInsertSide" \
    # "Soccer" \
    "button-press" \
    # #"pick-place" \
    # #"bin-picking" \
    # "button-press-topdown" \
    # "button-press-topdown-wall" \
    # #"door-lock" \
    # #"door-open" \
    # "door-unlock" \
    # #"drawer-close" \
    # "drawer-open" \
    # "faucet-close" \
    # #"faucet-open" \
    # #"handle-press" \
    # #"handle-pull" \
    # "handle-pull-side" \
    # #"lever-pull" \
    # "window-close" \
    # #"window-open" \
    )

factors=(
    "arm_pos" \
    # "camera_pos" \
    # # "distractor_pos"  \
    # "floor_texture" \
    # #"object_texture" \
    # "table_pos" \
    # "table_texture" \
)

#make the name of the dataset path
num_devices = nvidia-smi  -L | wc -l
n=0
for env in ${envs[@]}; do
    for factor in ${factors[@]}; do
        CUDA_VISIBLE_DEVICES=$n python train_rl_mw.py --config_path ../release/cfgs/metaworld/ibrl_basic.yaml  \
                                                      --bc_policy ${env}_${factor} \
                                                      --save_dir rl_models/metaworld/${env}_${factor}\
                                                      --use_wb 1 \
                                                      --wb_exp ${env}_rl_test \
                                                      --wb_run ${factor} \
                                                      --norm True
        n=$((n + 1))
        if (($n -eq $num_devices)); then
            echo waiting
            wait -n 
            n=0
        fi
    done 
done


