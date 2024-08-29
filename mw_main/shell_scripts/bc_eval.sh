
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

for env in ${envs[@]}; do
  python run_trained_policy.py --weight ./models/metaworld/${env}/model1.pt  --record_dir ./bc_eval/metaworld/${env} --seed 2024 --num_games 1
done