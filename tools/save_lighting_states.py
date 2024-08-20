import json
import argparse
#from common_utils import ibrl_utils as utils
from env.robosuite_wrapper import PixelRobosuite
from env.wrapper import LightingWrapper

def save_lighting_state(env_params, filename, num_states=10):

    #create an environment from a data file
    env = PixelRobosuite(**env_params)
    env = LightingWrapper(env, None)

    #wrap it in a lighting wrapper

    states = []
    #for some number of states, randomize the lighting wrapper
    for i in range(num_states):
        env.randomize()
        print(env.get_lighting_state())
        states.append(env.get_lighting_state())

    with open(filename, 'w') as fout:
        json.dump(states, fout)
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--num_states", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args()

    import train_rl
    agent, _, env_params = train_rl.load_model(args.weight, "cuda")
    save_lighting_state(env_params , args.output, args.num_states) 