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
    # "bin-picking" \
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
    "arm_pos" \
    "camera_pos" \
    # "distractor_pos"  \
    "floor_texture" \
    #"object_texture" \
    "table_pos" \
    "table_texture" \
    "light" \
    #"object_size"
)

#make the name of the dataset path
num_devices=`nvidia-smi  -L | wc -l`
n=0
for env in ${envs[@]}; do
    for factor in ${factors[@]}; do
        env CUDA_VISIBLE_DEVICES=$n \
        python train_bc_mw.py --dataset.path ${env}_${factor} --save_dir no_prop_bc_models/metaworld/${env}_${factor} --use_wb 1 --wb_exp ${env}_bc_no_prop --wb_run ${factor} --policy.use_prop 1 & #> log/metaworld/bc/${env}_${factor}.txt &
        n=$((n + 1))
        if test $n -eq $num_devices; then
            wait
            n=0
        fi
    done 
done


