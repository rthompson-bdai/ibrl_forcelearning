
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
    "drawer-open" \
    "faucet-close" \
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
    light \
)

num_devices=`nvidia-smi  -L | wc -l`
n=0

for env in ${envs[@]}; do
  for factor in ${factors[@]}; do
    env CUDA_VISIBLE_DEVICES=$n \
    python run_trained_policy.py --weight ./models/metaworld/${env}_${factor}/model1.pt  \
                                 --record_dir ./bc_eval_test/metaworld/${env}_${factor} \
                                 --seed 2024 \
                                 --num_games 50 \
                                 --eval &
                                #  > log/metaworld/bc/eval_${env}_${factor}.txt &

    n=$((n + 1))
    if test $n -eq $num_devices; then
        wait
        n=0
    fi
  done
done