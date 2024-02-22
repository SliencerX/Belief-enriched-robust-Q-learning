import random
import math
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random
import gym
import sys
from autoencoder_atari import *
from atari_utils import *
from wrappers import make_atari, wrap_deepmind, wrap_pytorch, make_atari_cart
from models import *
import cv2
from read_config import load_config
from argparser import argparser
import os
from utils import Logger, get_acrobot_eps, test_plot
def gen_traj_atari(encoder, autoencoder, traj_len=100, obs_num=5):

    #Pong
    env_params = {}
    env_params["crop_shift"] = 10
    env_params["restrict_actions"] = 4
    env_id = "PongNoFrameskip-v4"

    #Freeway
    # env_params = {}
    # env_params["crop_shift"] = 0
    # env_id = "FreewayNoFrameskip-v4"

    env = make_atari(env_id)
    env = wrap_deepmind(env, **env_params)
    env = wrap_pytorch(env)

    traj_ret = []
    obs = env.reset()
    obs = obs/255
    for _ in range(traj_len):
        #r_d = robbie.read_sensor(world, obs_num)
        step_data_raw = torch.tensor(obs).cuda().to(dtype=torch.float).unsqueeze(0)
        step_data_raw_noise = torch.tensor(np.clip(obs + np.random.uniform(-1/255, 1/255, size=(1,84,84)), 0 ,1)).cuda().to(dtype=torch.float).unsqueeze(0)
        #print(step_data_raw.shape)
        #print(encoder.features.cnn(step_data_raw).shape)
        step_data = autoencoder.encoder(encoder.features.cnn(step_data_raw)).squeeze(0).cpu().detach().numpy()
        r_d = autoencoder.encoder(encoder.features.cnn(step_data_raw_noise)).squeeze(0).cpu().detach().numpy()
        #action = env.action_space.sample()
        action = encoder.act(step_data_raw)[0]
        obs ,_ , done , _ = env.step(action)
        obs = obs/255

        # action = [d_x, d_y, d_h]
#        step_data = step_data + [action] + r_d
        step_data = np.concatenate((step_data,[action],r_d), axis = None)
        #print(step_data.shape)
        traj_ret.append(step_data)
        if done:
            obs = env.reset()
            obs = obs/255

    return np.array(traj_ret)

def gen_data_atari(args, num_trajs, traj_len=1000):
    config = load_config(args)
    prefix = config['env_id']
    training_config = config['training_config']
    test_config = config['test_config']
    attack_config = test_config["attack_config"]
    # if config['name_suffix']:
    #     prefix += config['name_suffix']
    # if config['path_prefix']:
    #     prefix = os.path.join(config['path_prefix'], prefix)
    # if 'load_model_path' in test_config and os.path.isfile(test_config['load_model_path']):
    #     if not os.path.exists(prefix):
    #         os.makedirs(prefix)
    #     test_log = os.path.join(prefix, test_config['log_name'])
    # else:
    #     if os.path.exists(prefix):
    #         test_log = os.path.join(prefix, test_config['log_name'])
    #     else:
    #         raise ValueError('Path {} not exists, please specify test model path.')
    # logger = Logger(open(test_log, "w"))
    # logger.log('Command line:', " ".join(sys.argv[:]))
    # logger.log(args)
    # logger.log(config)
    certify = test_config.get('certify', False)
    certify = test_config.get('certify', False)
    env_params = training_config['env_params']
    env_params['clip_rewards'] = False
    env_params['episode_life'] = False
    env_id = config['env_id']
    # env_params = training_config['env_params']
    # env_params['clip_rewards'] = False
    # env_params['episode_life'] = False
    # env_id = config['env_id']
    # mpl.rcParams['figure.figsize'] = (1,1)
    # mpl.rcParams['figure.dpi'] = 84
    # env_params = {}
    # env_params["crop_shift"] = 10
    # env_params["restrict_actions"] = 4
    # env_id = "PongNoFrameskip-v4"
    # env = gym.make(env_id)
    # env = make_env(env, frame_stack = False, scale = False)
    if "NoFrameskip" not in env_id:
        env = make_atari_cart(env_id)
    else:
        env = make_atari(env_id)
        env = wrap_deepmind(env, **env_params)
        env = wrap_pytorch(env)
    # env = make_atari(env_id)
    # env = wrap_deepmind(env, **env_params)
    # env = wrap_pytorch(env)
    model_width = training_config['model_width']
    robust_model = certify
    dueling = training_config.get('dueling', True)

    encoder = model_setup(env_id, env, robust_model, None, USE_CUDA, dueling, model_width)
    #encoder = model_setup(env_id, env, False, None, True, True, 1)
    #encoder.features.load_state_dict(torch.load("./PongNoFrameskip-v4/frame_4500000.pth"))
    #encoder.features.load_state_dict(torch.load("vanila_model.pth"))
    encoder.features.load_state_dict(torch.load("Pretrained/Pong-natural.model"))
    from tqdm import tqdm
    count = 0
    episode_reward = 0
    for _ in tqdm(range(num_trajs)):
        obs = env.reset()
        #obs = obs/255
        print(episode_reward)
        episode_reward = 0
        for _ in range(traj_len):
            obs_tensor = torch.from_numpy(np.ascontiguousarray(obs)).unsqueeze(0).cuda().to(torch.float32)
            obs_tensor /= 255
            action = encoder.act(obs_tensor)[0]
            #print(action)
            obs ,reward , done , _ = env.step(action)
            episode_reward += reward
            #print(episode_reward)
            #obs = obs/255
            #print(obs.shape)
            cv2.imwrite('./pong_pic/'+str(count)+'.png', obs.transpose(2,1,0))
            count += 1
            if done:
                obs = env.reset()

                #obs = obs/255
                break

if __name__ == "__main__":
    args=  argparser()
    gen_data_atari(args, 30)
