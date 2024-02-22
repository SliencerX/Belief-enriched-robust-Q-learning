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
from models_wocar import QNetwork, model_setup
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
from train_atari import *
from autoencoder_atari import *
import cv2
import csv

from paad_rl.a2c_ppo_acktr import algo, utils
from paad_rl.a2c_ppo_acktr.algo import gail
from paad_rl.a2c_ppo_acktr.arguments import get_args
from paad_rl.a2c_ppo_acktr.envs import make_vec_envs
from paad_rl.a2c_ppo_acktr.model import Policy
from paad_rl.a2c_ppo_acktr.storage import RolloutStorage
# from evaluation import evaluate
from paad_rl.attacker.attacker import *
from paad_rl.utils.dqn_core import DQN_Agent, Q_Atari,model_get
from paad_rl.utils.param import Param
from paad_rl.a2c_ppo_acktr.algo.kfac import KFACOptimizer

UINTS=[np.uint8, np.uint16, np.uint32, np.uint64]
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# def normalize_to_neg_one_to_one(img):
#     return img * 2 - 1
#
# def unnormalize_to_zero_to_one(t):
#     return (t + 1) * 0.5
#
# def sample_with_prior(diffusion, prior, steps = 10):
#     img = torch.from_numpy(np.asarray(prior)).unsqueeze(0).cuda().to(torch.float32)
#     tensor_steps = torch.linspace(1., 0., steps + 1).cuda().to(torch.float32)
#     for i in range(steps):
#         times = tensor_steps[i].cuda()
#         times_next = tensor_steps[i+1].cuda()
#         #print(*img.shape)
#         img = diffusion.p_sample(img, times, times_next)
#     img.clamp_(-1., 1.)
#     img = unnormalize_to_zero_to_one(img)
#     return img

def mini_test(env, model, diffusion, config, dtype, max_frames_per_episode=1000, seed = 1000):
    begin_time = time.time()
    device = torch.device('cuda')
    state = env.reset()
    all_rewards = []
    episode_reward = 0
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    state = env.reset()
    episode_idx = 1
    this_episode_frame = 1
    obs=[]
    pos=[]
    actions=[]
    # #import belief model
    # belief_args = parse_args()
    # #belief_model = get_model(belief_args).cuda()
    # belief_model = Belief_model(belief_args).cuda()
    # model_checkpoint, _ = get_checkpoint(belief_args)
    # # print(model_checkpoint)
    # belief_model.load_state_dict(model_checkpoint)
    # belief = []

    action_space = Box(-1.0, 1.0, (env.action_space.n-1,))
    pa_attacker = Policy(
        env.observation_space.shape,
        action_space,
        beta=False,
        epsilon=15/255,
        base_kwargs={'recurrent': False})
    if True:
        KFACOptimizer(pa_attacker) # the model structure for the acktr attacker is different
    old_steps, pa_attacker_state, _ = \
            torch.load('sa-dqn_models/paad/dqn_obspol_attacker_PongNoFrameskip-v4',  map_location=device)
    pa_attacker.load_state_dict(pa_attacker_state)
    pa_attacker.cuda()
    attacker = ObsPol_Attack(pa_attacker, det=True, cont=False)
    recurrent = torch.zeros(
        1, pa_attacker.recurrent_hidden_state_size).cuda()
    masks = torch.ones(1, 1).cuda()

    for frame_idx in range(1, max_frames_per_episode + 1):

        state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).cuda().to(torch.float32)/255
        # att_state = get_att_state(model, state/255, max_diff_norm = 1/255, sample_time = 10)
        # att_state_tensor = torch.from_numpy(np.ascontiguousarray(att_state)).unsqueeze(0).cuda().to(torch.float32)
        att_state_tensor = state_tensor


        # print("_______________att________________")
        # print("dis", np.linalg.norm(att_stat))
        # print("max",np.max(att_stat))
        # print("mean", np.mean(att_stat))
        # dif = restore_img[0].cpu().numpy() - restore_img_1[0].cpu().numpy()
        # print("_______________restore____________")
        # print("dis", np.linalg.norm(dif))
        # print("max",np.max(dif))
        # print("mean", np.mean(dif))


        # att_state_tensor = pgd(model, state_tensor, model.act(state_tensor), env_id = "Pong")
        # att_state = att_state_tensor.cpu().numpy()[0]

        # att_state_tensor = min_best(model, state_tensor, 15/255, pgd_steps=10, lr=1e-1, fgsm=False,
        #               norm=np.inf, rand_init=False, momentum=False, env_id = "Freeway")
        # att_state = att_state_tensor.cpu().numpy()[0]

        # att_state_tensor, recurrent = attacker.attack_torch(model, state_tensor, recurrent, masks, epsilon=15/255,
        #     fgsm=True, lr=0.1, pgd_steps=10, device=device)
        #print(att_state_tensor.shape)
        att_state = att_state_tensor.cpu().numpy()[0]

        #diffusion
        # att_state_diff = diffusion.p_sample_loop_with_prior((att_state.transpose(0,2,1)), samples = 1, steps = 20).cpu().numpy().transpose(0,1,3,2)
        # att_state_tensor = torch.from_numpy(np.ascontiguousarray(att_state)).unsqueeze(0).cuda().to(torch.float32)/255

        # cv2.imwrite('./test_pic_wocar/'+str(frame_idx)+'att.png', (att_state*255).transpose(2,1,0))
        # cv2.imwrite('./test_pic_wocar/'+str(frame_idx)+'ori.png', state.transpose(2,1,0))
        #cv2.imwrite('./test_pic/'+str(frame_idx)+'restore.png', (att_state_diff[0]*255).transpose(2,1,0))
        #restore_img = diffusion.p_sample_loop_with_prior((att_state.transpose(0,2      ,1)), steps = 250)
        #restore_img_1 = diffusion.p_sample_loop_with_prior((state.transpose(0,2,1)/255), steps = 250)
        #cv2.imwrite('./test_pic/'+str(frame_idx)+'restore_1.png', (restore_img_1[0].cpu().numpy()*255).transpose(1,2,0))
        #att_stat = att_state - state/255

        #print(att_state)
        #print(state_tensor, att_state_tensor)
        #belief = random_belief(att_state)
        #q = model.forward(att_state_tensor).detach().cpu().numpy()[0]
        #print(q)
        action = model.act(att_state_tensor, 0)[0]
        #print(action)
        #action = get_action_withbelief(env, model, att_state_diff, 0)
        #print(action)
        # if len(belief) == 0:
        #     action = model.act(att_state_tensor, 0)[0]
        # else:
        #     action = model.act(att_state_tensor, 0)[0]
            #action = get_action_withbelief(env, model, belief, 0)

        # obs.append(att_state)
        # temp_state = copy.deepcopy(state)
        # temp_state = list(temp_state)
        # temp_state = np.asarray(temp_state)
        # pos.append(temp_state)
        # actions.append([action])
        # obs_tensor = torch.Tensor(np.asarray(obs)).unsqueeze(0).cuda()
        # pos_tensor = torch.Tensor(np.asarray(pos)).unsqueeze(0).cuda()
        # actions_tensor = torch.Tensor(np.asarray(actions)).unsqueeze(0).cuda()

        #belief = get_belief(belief_model, belief_args, env_map, obs_tensor, pos_tensor, actions_tensor)
        #belief = []
        #action = model.act(state_tensor)[0]
        next_state, reward, done, _ = env.step(action)
        # logger.log(action)
        state = next_state
        episode_reward += reward
        if frame_idx%50 == 0:
            print(frame_idx, episode_reward)
        if this_episode_frame == max_frames_per_episode:
            done = True
        if done:
            break
        else:
            this_episode_frame += 1
    duration = time.time()-begin_time
    fps = frame_idx/duration
    # print(duration)
    # print("FPS", frame_idx/duration)
    return episode_reward, fps

def main(args):
    model = Unet(
        dim = 64,
        dim_mults = (1, 2),
        channels = 1
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 84,
        # channels = 1,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l1'            # L1 or L2
    )
    #torch.load(diffusion, "./results/model-150.pt")
    # diffusion.load_state_dict(torch.load("./results/model-150.pt")['model'])
    # diffusion = diffusion.cuda()
    diffusion = None
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
    if "NoFrameskip" not in env_id:
        env = make_atari_cart(env_id)
    else:
        env = make_atari(env_id)
        env = wrap_deepmind(env, **env_params)
        env = wrap_pytorch(env)
    state = env.reset()
    dtype = state.dtype

    model_width = training_config['model_width']
    certify = test_config.get('certify', False)
    robust_model = certify
    dueling = training_config.get('dueling', True)


    model = model_setup(env_id, env, robust_model, None, USE_CUDA, dueling, model_width)
    #model_path = "Grid_Continous/att_frame_890000.pth"
    #model_path = "PongNoFrameskip-v4_very_good/att_frame_370000.pth"
    #model_path = "PongNoFrameskip-v4/frame_3200000.pth"
    #model_path = "PongNoFrameskip-v4_good/att_frame_170000.pth"
    #model_path = "sa-dqn_models/models/Pong-convex-3.pth"
    model_path = "sa-dqn_models/wocar_models/Pong-wocar-pgd.pth"
    #model_path = "sa-dqn_models/wocar_models/frame_3110000.pth"
    #model_path = "sa-dqn_models/wocar_models/Freeway-wocar-pgd.pth"


    model.features.load_state_dict(torch.load(model_path))

    # seed = training_config['seed']
    # env.seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed)
    seed = 5004
    reward  = []
    fps = []
    for i in range(5):
        tmp, tmp1 = mini_test(env, model, diffusion, config, dtype, max_frames_per_episode=5000, seed = seed+i)
        print(seed+i, tmp)
        reward.append(tmp)
        fps.append(tmp1)
    # print("average reward is ", np.mean(reward))
    # file = open("paad-15-255-freeway-wocar.csv", 'w')
    # writer = csv.writer(file)
    # writer.writerow(reward)
    # file.close()
    file = open("Pong-wocar-fps.csv", 'w')
    writer = csv.writer(file)
    writer.writerow(fps)
    file.close()

if __name__ == "__main__":
    args=  argparser()
    main(args)
