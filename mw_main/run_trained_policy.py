from train_bc_mw import load_model as bc_load_model
from train_rl_mw import load_model as rl_load_model
from eval_mw import run_eval
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--mode", type=str, default="bc", help="bc/rl")
    parser.add_argument("--num_games", type=int, default=10)
    parser.add_argument("--record_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--save_plots", action="store_true")
    args = parser.parse_args()

    if args.mode == 'bc':
        policy, env, env_params = bc_load_model(args.weight, 'cuda', eval=args.eval)
    else:
        policy, env, env_params = rl_load_model(args.weight, 'cuda', eval=args.eval)
    scores = run_eval(env, policy, num_game=args.num_games, seed=args.seed, record_dir=args.record_dir, verbose=False, save_video=args.save_video, save_plots=args.save_plots, save_images=args.save_images)