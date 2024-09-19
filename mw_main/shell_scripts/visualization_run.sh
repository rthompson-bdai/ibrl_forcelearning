
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

num_devices=`nvidia-smi  -L | wc -l`
n=0

for env in ${envs[@]}; do
    for factor in ${factors[@]}; do
        env CUDA_VISIBLE_DEVICES=$n \
        python run_trained_policy.py --weight ./no_prop_bc_models/metaworld/${env}_${factor}/model1.pt  \
                                    --record_dir ./bc_viz_force/metaworld/${env}_${factor} \
                                    --mode bc \
                                    --seed 2024 \
                                    --save_video \
                                    --save_images \
                                    --num_games 10 \
                                    &
        n=$((n + 1))
        if test $n -eq $num_devices; then
            wait
            n=0
        fi
    done
done

for env in ${envs[@]}; do
    for factor in ${factors[@]}; do
        env CUDA_VISIBLE_DEVICES=$n \
        python run_trained_policy.py --weight ./models/metaworld/${env}_${factor}/model1.pt  \
                                    --record_dir ./bc_viz_no_force/metaworld/${env}_${factor} \
                                    --mode bc \
                                    --seed 2024 \
                                    --save_video \
                                    --save_images \
                                    --num_games 10 \
                                    &
        n=$((n + 1))
        if test $n -eq $num_devices; then
            wait
            n=0
        fi
    done
done



# for env in ${envs[@]}; do
#     for factor in ${factors[@]}; do
#         env CUDA_VISIBLE_DEVICES=$n \
#         python run_trained_policy.py --weight ./no_prop_rl_models/metaworld/2024/${env}_${factor}_no_force/model0.pt  \
#                                     --record_dir ./rl_viz_no_force/metaworld/${env}_${factor} \
#                                     --mode rl \
#                                     --seed 2024 \
#                                     --save_video \
#                                     --save_images \
#                                     --num_games 10 \
#                                     &
#         n=$((n + 1))
#         if test $n -eq $num_devices; then
#             wait
#             n=0
#         fi
#     done
# done
