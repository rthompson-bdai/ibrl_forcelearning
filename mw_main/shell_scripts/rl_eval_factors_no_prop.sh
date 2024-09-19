
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
seeds=( 2024 2025 2026 2027 2028 )

# for seed in ${seeds[@]}; do
#     for env in ${envs[@]}; do
#         for factor in ${factors[@]}; do
#             env CUDA_VISIBLE_DEVICES=$n \
#             python run_trained_policy.py --weight ./no_prop_rl_models/metaworld/${seed}/${env}_${factor}_no_force/model0.pt  \
#                                         --record_dir ./rl_eval_test_no_prop_no_force/metaworld/${seed}/${env}_${factor} \
#                                         --mode rl \
#                                         --seed ${seed} \
#                                         --num_games 50 \
#                                         --eval &
#             n=$((n + 1))
#             if test $n -eq $num_devices; then
#                 wait
#                 n=0
#             fi
#         done
#     done
# done

# for seed in ${seeds[@]}; do
#     for env in ${envs[@]}; do
#         for factor in ${factors[@]}; do
#             env CUDA_VISIBLE_DEVICES=$n \
#             python run_trained_policy.py --weight ./no_prop_rl_models/metaworld/${seed}/${env}_${factor}_force/model0.pt  \
#                                         --record_dir ./rl_eval_test_no_prop_force/metaworld/${seed}/${env}_${factor} \
#                                         --mode rl \
#                                         --seed ${seed} \
#                                         --num_games 50 \
#                                         --eval &
#             n=$((n + 1))
#             if test $n -eq $num_devices; then
#                 wait
#                 n=0
#             fi
#         done
#     done
# done

# for seed in ${seeds[@]}; do
#     for env in ${envs[@]}; do
#         for factor in ${factors[@]}; do
#             env CUDA_VISIBLE_DEVICES=$n \
#             python run_trained_policy.py --weight ./no_prop_rl_models/metaworld/${seed}/${env}_${factor}_no_force/model0.pt  \
#                                         --record_dir ./rl_eval_train_no_prop_no_force/metaworld/${seed}/${env}_${factor} \
#                                         --mode rl \
#                                         --seed ${seed} \
#                                         --num_games 50 &
#             n=$((n + 1))
#             if test $n -eq $num_devices; then
#                 wait 
#                 n=0
#             fi
#         done
#     done
# done

for seed in ${seeds[@]}; do
    for env in ${envs[@]}; do
        for factor in ${factors[@]}; do
            env CUDA_VISIBLE_DEVICES=$n \
            python run_trained_policy.py --weight ./no_prop_rl_models/metaworld/${seed}/${env}_${factor}_force/model0.pt  \
                                        --record_dir ./rl_eval_train_no_prop_force/metaworld/${seed}/${env}_${factor} \
                                        --mode rl \
                                        --seed ${seed} \
                                        --num_games 50 &
            n=$((n + 1))
            if test $n -eq $num_devices; then
                wait
                n=0
            fi
        done
    done
done

for seed in ${seeds[@]}; do
    for env in ${envs[@]}; do
        for factor in ${factors[@]}; do
            env CUDA_VISIBLE_DEVICES=$n \
            python run_trained_policy.py --weight ./no_prop_rl_models/metaworld/${seed}/${env}_${factor}_force/model0.pt  \
                                        --record_dir ./rl_eval_test_no_prop_force/metaworld/${seed}/${env}_${factor} \
                                        --mode rl \
                                        --seed ${seed} \
                                        --num_games 50 \
                                        --eval &
            n=$((n + 1))
            if test $n -eq $num_devices; then
                wait
                n=0
            fi
        done
    done
done

for seed in ${seeds[@]}; do
    for env in ${envs[@]}; do
        for factor in ${factors[@]}; do
            env CUDA_VISIBLE_DEVICES=$n \
            python run_trained_policy.py --weight ./norm_warmup_no_prop_rl_models/metaworld/${seed}/${env}_${factor}_force/model0.pt  \
                                        --record_dir ./norm_warmup_no_prop_force_train/metaworld/${seed}/${env}_${factor} \
                                        --mode rl \
                                        --seed ${seed} \
                                        --num_games 50 &
            n=$((n + 1))
            if test $n -eq $num_devices; then
                wait 
                n=0
            fi
        done
    done
done

for seed in ${seeds[@]}; do
    for env in ${envs[@]}; do
        for factor in ${factors[@]}; do
            env CUDA_VISIBLE_DEVICES=$n \
            python run_trained_policy.py --weight ./norm_warmup_no_prop_rl_models/metaworld/${seed}/${env}_${factor}_force/model0.pt  \
                                        --record_dir ./norm_warmup_no_prop_force_test/metaworld/${seed}/${env}_${factor} \
                                        --mode rl \
                                        --seed ${seed} \
                                        --num_games 50 \
                                        --eval &
            n=$((n + 1))
            if test $n -eq $num_devices; then
                wait 
                n=0
            fi
        done
    done
done

#for plots

# for seed in ${seeds[@]}; do
#     for env in ${envs[@]}; do
#         for factor in ${factors[@]}; do
#             env CUDA_VISIBLE_DEVICES=$n \
#             python run_trained_policy.py --weight ./norm_warmup_no_prop_rl_models/metaworld/${seed}/${env}_${factor}_force/model0.pt  \
#                                         --record_dir ./norm_warmup_no_prop_force_graphs/metaworld/${seed}/${env}_${factor} \
#                                         --mode rl \
#                                         --seed ${seed} \
#                                         --save_video \
#                                         --save_plots \
#                                         --num_games 10 &
#             n=$((n + 1))
#             if test $n -eq $num_devices; then
#                 wait 
#                 n=0
#             fi
#         done
#     done
# done