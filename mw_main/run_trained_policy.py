from train_bc_mw import load_model
from eval_mw import run_eval
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--mode", type=str, default="bc", help="bc/rl")
    parser.add_argument("--num_games", type=int, default=10)
    parser.add_argument("--record_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    policy, env, env_params = load_model(args.weight, 'cuda')
    scores = run_eval(env, policy, num_game=args.num_games, seed=args.seed, record_dir=args.record_dir, verbose=False)