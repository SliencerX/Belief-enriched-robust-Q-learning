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
from models import QNetwork, model_setup
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
import importlib

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

from diffusion_distiller.train_utils import *
from diffusion_distiller.v_diffusion import make_beta_schedule


UINTS=[np.uint8, np.uint16, np.uint32, np.uint64]
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

def random_belief(att_state, max_diff_norm = 1/255):
    random_noise = np.random.uniform(-max_diff_norm, max_diff_norm, 28224).reshape(4,1,84,84)
    return random_noise+att_state

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
            torch.load('sa-dqn_models/paad/dqn_obspol_attacker_BankHeistNoFrameskip-v4',  map_location=device)
    pa_attacker.load_state_dict(pa_attacker_state)
    pa_attacker.cuda()
    attacker = ObsPol_Attack(pa_attacker, det=True, cont=False)
    recurrent = torch.zeros(
        1, pa_attacker.recurrent_hidden_state_size).cuda()
    masks = torch.ones(1, 1).cuda()
    #print("training steps for this model:", old_steps)


    begin_time = time.time()
    for frame_idx in range(1, max_frames_per_episode + 1):

        state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).cuda().to(torch.float32)/255
        # att_state = get_att_state(model, state/255, max_diff_norm = 1/255, sample_time = 10)
        # att_state_tensor = torch.from_numpy(np.ascontiguousarray(att_state)).unsqueeze(0).cuda().to(torch.float32)


        # print("_______________att________________")
        # print("dis", np.linalg.norm(att_stat))
        # print("max",np.max(att_stat))
        # print("mean", np.mean(att_stat))
        # dif = restore_img[0].cpu().numpy() - restore_img_1[0].cpu().numpy()
        # print("_______________restore____________")
        # print("dis", np.linalg.norm(dif))
        # print("max",np.max(dif))
        # print("mean", np.mean(dif))


        #att_state_tensor = state_tensor
        #att_state_tensor = pgd(model, state_tensor, model.act(state_tensor), env_id = "Pong")

        # att_state_tensor = min_best(model, state_tensor, 15/255, pgd_steps=10, lr=1e-1, fgsm=False,
        #                 norm=np.inf, rand_init=False, momentum=False, env_id = "Freeway")
        #
        att_state_tensor, recurrent = attacker.attack_torch(model, torch.from_numpy(np.ascontiguousarray(state)).cuda().to(torch.float32)/255, recurrent, masks, epsilon= 15/255,
            fgsm=True, lr=0.1, pgd_steps=10, device=device)
        #print(att_state_tensor.shape)
        att_state = att_state_tensor.cpu().numpy()[0]

        #diffusion
        # att_state_diff = diffusion.p_sample_loop_with_prior((att_state.transpose(0,2,1)), samples = 1, steps = 10).cpu().numpy().transpose(0,1,3,2)

        # for _ in range(1):
        #      att_state = np.clip((np.random.uniform(-35/255, 35/255, size = (1,84,84)) + att_state), 0 ,1)

        for _ in range(1):
             att_state = np.clip((np.random.normal(0, 1/255, size = (1,84,84)) + att_state), 0 ,1)

        att_state_diff = make_visualization_withprior(diffusion, device, [1, 1, 84, 84], att_state.transpose(0,2,1), need_tqdm=False, eta=0, clip_value=1.2).cpu().numpy().transpose(0,1,3,2)
        # for _ in range(2):
        #     att_state_diff = np.clip((np.random.uniform(-1/255, 1/255, size = (2,1,84,84)) + att_state_diff), 0 ,1)
        #att_state = att_state_tensor.squeeze(0).cpu().numpy()
        #att_state_tensor = torch.from_numpy(np.ascontiguousarray(att_state)).unsqueeze(0).cuda().to(torch.float32)/255
        #att_state_tensor = torch.from_numpy(np.ascontiguousarray(att_state_diff[0])).unsqueeze(0).cuda().to(torch.float32)
        #att_state_diff = random_belief(att_state, max_diff_norm = 15/255)
        # #
        cv2.imwrite('./test_pic/'+str(frame_idx)+'att.png', (att_state*255).transpose(2,1,0))
        # cv2.imwrite('./test_pic/'+str(frame_idx)+'ori.png', state.transpose(2,1,0))
        cv2.imwrite('./test_pic/'+str(frame_idx)+'restore.png', (att_state_diff[0]*255).transpose(2,1,0))
        #restore_img = diffusion.p_sample_loop_with_prior((att_state.transpose(0,2      ,1)), steps = 250)
        #restore_img_1 = diffusion.p_sample_loop_with_prior((state.transpose(0,2,1)/255), samples = 1, steps = 12).cpu().numpy().transpose(0,1,3,2)
        #cv2.imwrite('./test_pic/'+str(frame_idx)+'restore_1.png', (restore_img_1[0]*255).transpose(2,1,0))
        #att_stat = att_state - state/255
        # print(np.linalg.norm(att_state - state/255))
        # print(np.linalg.norm(att_state_diff[0] - restore_img_1[0]))

        #print(att_state)
        #print(state_tensor, att_state_tensor)
        #att_state_diff = random_belief(att_state)
        #print(model.forward(att_state_tensor))
        #action = model.act(att_state_tensor, 0)[0]
        #print(action)
        action = get_action_withbelief(env, model, att_state_diff, 0)
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

    M = importlib.import_module('diffusion_distiller.atari_u')
    make_model = getattr(M, "make_model")
    device = torch.device("cuda")
    teacher_ema = make_model().to(device)
    def make_diffusion(args, model, n_timestep, time_scale, device):
        betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
        M = importlib.import_module("diffusion_distiller.v_diffusion")
        D = getattr(M, "GaussianDiffusion")
        sampler = "ddpm"
        if False:
            sampler = "clipped"
        return D(model, betas, time_scale=time_scale, sampler=sampler)
    teacher = make_model().to(device)
    ckpt = torch.load('./diffusion_distiller/checkpoints/atari_bank/base_3/checkpoint.pt')
    teacher.load_state_dict(ckpt["G"])
    n_timesteps = ckpt["n_timesteps"]
    time_scale = ckpt["time_scale"]
    del ckpt
    print("Model loaded.")
    teacher_diffusion = make_diffusion(args, teacher, n_timesteps, time_scale, device)
    # image_size[0] = args.batch_size
    teacher_diffusion.num_timesteps = 2


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
    #model_path = "FreewayNoFrameskip-v4/att_frame_350000.pth"
    #model_path = "FreewayNoFrameskip-v4_good/att_frame_535000.pth"
    #model_path = "PongNoFrameskip-v4/frame_3000000.pth"
    #model_path = "sa-dqn_models/models/Pong-natural.model"
    #model_path = "FreewayNoFrameskip-v4/frame_2400000.pth"
    #model_path = "FreewayNoFrameskip-v4_tmp/att_frame_2000000.pth"
    #model_path = "FreewayNoFrameskip-v4_ran/att_frame_155000.pth"
    #model_path = "PongNoFrameskip-v4_ran_tmp/att_frame_500000.pth"
    #model_path = "PongNoFrameskip-v4_ran_good/att_frame_270000.pth"
    #model_path = "PongNoFrameskip-v4/att_frame_880000.pth"
    model_path = "BankHeistNoFrameskip-v4_ori/frame_5800000.pth"

    model.features.load_state_dict(torch.load(model_path))

    # seed = training_config['seed']
    # env.seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed)
    #seed = 5007
    seed = 1000
    reward  = []
    fps = []
    for i in range(5):
        tmp, tmp1 = mini_test(env, model, teacher_diffusion, config, dtype, max_frames_per_episode=5000, seed = seed+i*100)
        print(seed+i, tmp)
        reward.append(tmp)
        fps.append(tmp1)
    print("average reward is ", np.mean(reward))
    print(reward)
    # file = open("vanila_pong_fast.csv", 'w')
    # writer = csv.writer(file)
    # writer.writerow(reward)
    # file.close()
    # file = open("vanila_pong_fast.csv", 'w')
    # writer = csv.writer(file)
    # writer.writerow(fps)
    # file.close()

if __name__ == "__main__":
    args=  argparser()
    main(args)
