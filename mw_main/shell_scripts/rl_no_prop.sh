
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
    "arm_pos" \
    "camera_pos" \
    #"distractor_pos"  \
    "floor_texture" \
    #"object_texture" \
    "table_pos" \
    "table_texture" \
    "light" \
)

#make the name of the dataset path

seeds=( 2026 2027 2028 ) #2024 2025 
num_devices=`nvidia-smi  -L | wc -l`
n=0

# for seed in ${seeds[@]}; do
#     for env in ${envs[@]}; do
#         for factor in ${factors[@]}; do
#             env CUDA_VISIBLE_DEVICES=$n \
#             python train_rl_mw.py --config_path ../release/cfgs/metaworld/ibrl_basic_force_only.yaml  \
#                                                         --bc_policy ${env}_${factor} \
#                                                         --save_dir no_prop_rl_models/metaworld/${seed}/${env}_${factor}_force \
#                                                         --use_wb 1 \
#                                                         --wb_exp no_prop_${factor}_rl \
#                                                         --wb_run ${env}_force_${seed} \
#                                                         --seed ${seed} \
#                                                         --no_prop True \
#                                                         --norm True \
#                                                         &> log/rl_force_only_${env}_${factor}.txt &
#             n=$((n + 1))
#             if test $n -eq $num_devices; then
#                 wait
#                 n=0
#             fi
#         done 
#     done
# done

num_devices=`nvidia-smi  -L | wc -l`
n=0
for seed in ${seeds[@]}; do
    for env in ${envs[@]}; do
        for factor in ${factors[@]}; do
            env CUDA_VISIBLE_DEVICES=$n python train_rl_mw.py --config_path ../release/cfgs/metaworld/ibrl_basic_no_prop.yaml  \
                                                        --bc_policy ${env}_${factor} \
                                                        --save_dir no_prop_rl_models/metaworld/${seed}/${env}_${factor}_no_force\
                                                        --use_wb 1 \
                                                        --seed ${seed} \
                                                        --no_prop True \
                                                        --wb_exp no_prop_${factor}_rl \
                                                        --wb_run ${env}_no_force_${seed} \
                                                        &> log/rl_no_prop_${env}_${factor}.txt &
            n=$((n + 1))
            if test $n -eq $num_devices; then
                wait
                n=0
            fi
        done 
    done
done


