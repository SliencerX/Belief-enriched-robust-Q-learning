import sys
sys.path.append("./common")
sys.path.append("./auto_LiRPA")
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from argparser import argparser
from eps_scheduler import EpsilonScheduler
from read_config import load_config
import numpy as np
import cpprb
import re
from attacks import attack
import gym
import random
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch, make_atari_cart
import torch.optim as optim
import torch
from torch.nn import CrossEntropyLoss
import torch.autograd as autograd
import math
import time
import os
import argparse
import copy
from datetime import datetime
from utils import CudaTensorManager, ActEpsilonScheduler, BufferBetaScheduler, Logger, update_target, get_acrobot_eps, plot
from my_replay_buffer import ReplayBuffer, NaivePrioritizedBuffer
from common.replay_buffer import PrioritizedReplayBuffer
from async_env import AsyncEnv
from async_rb import AsyncReplayBuffer
from gridworld import *
from ibp import *
from data_utils import *
from pfrnns.model import *
from pfrnns.evaluate import *
from train import *
import csv

UINTS=[np.uint8, np.uint16, np.uint32, np.uint64]
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class DuelingDQN(nn.Module):
    def __init__(self, num_channels, action_space):
        super(DuelingDQN, self).__init__()
        self.num_actions = action_space.n

        self.advantage = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        #x = x.squeeze(0)
        advantage = self.advantage(x)
        value = self.value(x)
        return value, advantage - torch.mean(advantage, dim=1, keepdim=True)

    def act(self, state, epsilon):
        with torch.no_grad():
            if random.random() > epsilon:
                value, advs = self.forward(state)
                q_values = value + advs
                #print(q_value)
                action  = torch.argmax(q_values, dim=1)[0]
            else:
                action = random.randrange(self.num_actions)
        return action


def mini_test(model, config, dtype, num_episodes=10, max_frames_per_episode=100):

    env = Gridmaze()
    state = env.reset()
    all_rewards = []
    all_fps = []
    episode_reward = 0
    seed = 5000
    env.seed(seed)
    state = env.reset()
    episode_idx = 1
    this_episode_frame = 1
    obs=[]
    pos=[]
    actions=[]
    env_map = np.loadtxt('maze.csv', delimiter=',')
    map_mean = np.mean(env_map)
    map_std = np.std(env_map)
    env_map = torch.FloatTensor(env_map).unsqueeze(0)
    env_map = ((env_map - map_mean) / map_std).unsqueeze(0).cuda()
    #import belief model
    belief_args = parse_args()
    #print(belief_args)
    belief_model = get_model(belief_args).cuda()
    model_checkpoint, _ = get_checkpoint(belief_args)
    #print(model_checkpoint)
    belief_model.load_state_dict(model_checkpoint)
    belief = []
    begin_time = time.time()
    begin_time_1 = time.time()
    frame_count = 0
    for frame_idx in range(1, num_episodes * max_frames_per_episode + 1):
        frame_count+=1
        #print(frame_idx)
        state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).cuda().to(torch.float32)
        # att_state = get_att_state(model, state, sample_time = 100)
        # att_state_tensor = torch.from_numpy(np.ascontiguousarray(att_state)).unsqueeze(0).cuda().to(torch.float32)
        #att_state_tensor = pgd(model, state_tensor, model.act(state_tensor, 0), env_id = "Grid")
        # att_state = att_state_tensor.cpu().numpy()[0]

        att_state_tensor = min_best(model, state_tensor, 0.5, pgd_steps=10, lr=1e-1, fgsm=False,
                    norm=np.inf, rand_init=False, momentum=False, env_id = "Grid")
        #att_state_tensor = state_tensor


        att_state = att_state_tensor.cpu().numpy()[0]
        #print(att_state)
        #print(state_tensor, att_state_tensor)
        #belief = random_belief(att_state)
        #if len(belief) == 0:
        if True:
            #belief = random_belief(att_state)
            #print(belief[0])
            #action = get_action_withbelief(env, model, [belief[0]], 0)
            action = model.act(att_state_tensor, 0)
        else:
            #action = model.act(att_state_tensor, 0)[0]
            action = get_action_withbelief(env, model, [belief[0]], 0)

        obs.append(att_state)
        h = env.robbie.h / 360 * 2 * np.pi
        # temp_state = copy.deepcopy(state)
        # temp_state = list(temp_state)
        # temp_state.append(h)
        # temp_state = np.asarray(temp_state)
        # pos.append(temp_state)
        # actions.append([action])
        # obs_tensor = torch.Tensor(np.asarray(obs)).unsqueeze(0).cuda()
        # pos_tensor = torch.Tensor(np.asarray(pos)).unsqueeze(0).cuda()
        # actions_tensor = torch.Tensor(np.asarray(actions)).unsqueeze(0).cuda()

        #belief = get_belief(belief_model, belief_args, env_map, obs_tensor, pos_tensor, actions_tensor)

        #action = model.act(state_tensor)[0]
        next_state, reward, done, _ = env.step(action)
        # logger.log(action)
        state = next_state
        episode_reward += reward
        if this_episode_frame == max_frames_per_episode:
            done = True
        if done:
            end_time = time.time()
            fps = frame_count/(end_time - begin_time)
            obs = []
            pos = []
            actions = []
            state = env.reset()
            print(episode_idx, episode_reward)
            all_rewards.append(episode_reward)
            all_fps.append(fps)
            episode_reward = 0
            this_episode_frame = 1
            episode_idx += 1
            if episode_idx > num_episodes:
                break
            begin_time = time.time()
            frame_count = 0
        else:
            this_episode_frame += 1
            frame_count+=1
    end_time = time.time()
    print(frame_idx/(end_time - begin_time_1))
    return all_rewards, all_fps

def main(args):
    config = load_config(args)
    prefix = config['env_id']
    training_config = config['training_config']
    test_config = config['test_config']
    attack_config = test_config["attack_config"]
    if config['name_suffix']:
        prefix += config['name_suffix']
    if config['path_prefix']:
       prefix = os.path.join(config['path_prefix'], prefix)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    env_params = training_config['env_params']
    env_id = config['env_id']
    env = Gridmaze()
    state = env.reset()
    dtype = state.dtype

    model_width = training_config['model_width']
    certify = test_config.get('certify', False)
    robust_model = certify
    dueling = training_config.get('dueling', True)


    #model = model_setup(env_id, env, robust_model, None, USE_CUDA, dueling, model_width)
    model = DuelingDQN(env.observation_space.shape[0], env.action_space)
    #model_path = "Grid_Continous_good/att_frame_890000.pth"
    #model_path = "Grid_Continous_good/frame_800000.pth"
    #model_path = "Grid_Continous/att_conservative_frame_495000.pth"
    #model_path = "Grid_Continous_sa_dqn/frame_1000000.pth"
    #model_path = "Grid_Continous/frame_1000000.pth"
    model_path = "sa-dqn_models/radial_models/Grid.pt"

    saved_state = torch.load(
        model_path,
        map_location=lambda storage, loc: storage)
    if "model_state_dict" in saved_state.keys():
        saved_state = saved_state['model_state_dict']
    model.load_state_dict(saved_state)
    model.cuda()

    seed = 1000
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    reward, fps = mini_test(model, config, dtype, num_episodes=10, max_frames_per_episode=100)
    file = open("minbest-0.5-gridmaze-radial.csv", 'w')
    writer = csv.writer(file)
    writer.writerow(reward)
    file.close()
    print("average reward is ", np.mean(reward))

    file = open("minbest-gridmaze-radial-fps.csv", 'w')
    writer = csv.writer(file)
    writer.writerow(fps)
    file.close()
    #print("average reward is ", np.mean(reward))

if __name__ == "__main__":
    args=  argparser()
    main(args)
