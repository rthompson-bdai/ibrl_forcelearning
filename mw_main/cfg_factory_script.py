import yaml 


#makes a bunch of rl training configs for different factors

#provide the environment name
#provide the factors


config_dict = {
'episode_length': 100,
'stddev_max': 0.1,
'bc_policy': args.env_file,
'factors': args.factors
'preload_num_data': 3,
'env_reward_scale': 1,
'num_train_step': 100000,
'replay_buffer_size': 500,
'use_wb': 1,
"wb_exp": "metaworld_rl"
"wb_group": f"{args.env}_{"_".join(args.factors)}",
'num_eval_episode': 20,
'q_agent': {
  'use_prop': True
  'act_method': "ibrl",
  'enc_type': "drq",
  'actor':{
    'dropout': 0,
    'hidden_dim': 1024,
    'feature_dim': 64,
  },
  'critic':
    {'feature_dim': 64}
}
}
