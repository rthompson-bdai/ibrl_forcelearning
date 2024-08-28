
#!/bin/bash

envs=( #"Assembly" \
    # "Basketball" \
    # "CoffeePush" \
    # "BoxClose" \
    # "StickPull" \
    # "PegInsertSide" \
    # "Soccer" \
    "button-press" \
    # "pick-place" \
    # "bin-picking" \
    # "button-press-topdown" \
    # "button-press-topdown-wall" \
    # "door-lock" \
    # "door-open" \
    # "door-unlock" \
    # "drawer-close" \
    # "drawer-open" \
    # "faucet-close" \
    # "faucet-open" \
    # "handle-press" \
    # "handle-pull" \
    # "handle-pull-side" \
    # "lever-pull" \
    # "window-close" \
    # "window-open" \
)

factors=(
    "arm_pos" \
    "camera_pos" \
    #"distractor_pos"  \
    "floor_texture" \
    #"object_texture" \
    "table_pos" \
    "table_texture" \
)

for env in ${envs[@]}; do
  for factor in ${factors[@]}; do
    python run_trained_policy.py --weight ./models/metaworld/${env}_${factor}/model1.pt  --record_dir ./bc_eval/metaworld/${env}_${factor} --seed 2024 --num_games 5
  done
done