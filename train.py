import sys
sys.path.append("./common")
sys.path.append("./auto_LiRPA")
sys.path.append("./pfrnns")
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


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
UINTS=[np.uint8, np.uint16, np.uint32, np.uint64]

def linear_schedule_exp(min_exp, ini_exp, total, current):
    if current >= total:
        return min_exp
    return max(min_exp, ((total-current)*ini_exp)/total)

def get_belief(model, args, env_map, obs, pos, actions):
    model.eval()
    # print(obs.shape)
    # print(pos.shape)
    # print(actions.shape)
    with torch.no_grad():
        model.zero_grad()
        loss, log_loss, particle_pred = model.step(
            env_map, obs, actions, pos, args)
        temp = particle_pred.squeeze(1)
        last = temp[:,-1,:]
        result = []
        for i in last:
            x = i.data.cpu().numpy()[0]*10
            y = i.data.cpu().numpy()[1]*10
            result.append(np.asarray([x,y]))
    return np.asarray(result)

def random_belief(att_state, max_diff_norm = 0.5):
    random_noise = np.random.uniform(-max_diff_norm, max_diff_norm, 60).reshape(30,2)
    return random_noise+att_state


def move_belief(env, action, belief):
    new_belief = []
    dummy_robbie = Robot(env.world, speed=env.speed)
    for b in belief:
        dummy_robbie.x = b[0]
        dummy_robbie.y = b[1]
        dummy_robbie.h = env.direction[action]
        dummy_robbie.move_one(env.world)
        new_belief.append(np.asaray[dummy_robbie.x, dummy_robbie.y])
    return np.asarray(new_belief)

def get_action_withbelief(env, model, belief, eps):
    value = dict()
    for action in range(len(env.direction)):
        temp = []
        tensor_states = []
        for b in belief:
            state_tensor = torch.tensor(list(b)).cuda()
            tensor_states.append(state_tensor)
        tensor_states = torch.stack(tensor_states).cuda().to(torch.float32)
        #print(tensor_states)
        #tensor_state = torch.from_numpy(np.ascontiguousarray(b)).unsqueeze(0).cuda().to(torch.float32)
        #print(np.min(model.forward(tensor_states)[:,action].data.cpu().numpy()))
        #temp.append(model.forward(tensor_states)[0][action].data.cpu())
        #print(temp)
        value[action]=np.min(model.forward(tensor_states)[:,action].data.cpu().numpy())
    action = (max(value, key = value.get))
    mask = np.random.choice(np.arange(0, 2), p=[1-eps, eps])
    action = (1-mask) * action + mask * np.random.randint(env.action_space.n)
    return action


def get_logits_lower_bound(model, state, state_ub, state_lb, eps, C, beta):
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps, x_L=state_lb, x_U=state_ub)
    bnd_state = BoundedTensor(state, ptb)
    pred = model.features(bnd_state, method_opt="forward")
    logits_ilb, _ = model.features.compute_bounds(C=C, IBP=True, method=None)
    if beta < 1e-5:
        logits_lb = logits_ilb
    else:
        logits_clb, _ = model.features.compute_bounds(IBP=False, C=C, method="backward", bound_upper=False)
        logits_lb = beta * logits_clb + (1-beta) * logits_ilb
    return logits_lb

def get_att_state(model, true_state, max_diff_norm = 0.5, sample_time = 10):
    possible_action = set()
    d = model.env.observation_space.shape[0]
    values = dict()
    tensor_true_state = torch.from_numpy(np.ascontiguousarray(true_state)).unsqueeze(0).cuda().to(torch.float32)
    tensor_purturb_states = []
    purturb_states = []
    # for _ in range(sample_time):
    #     x = np.random.uniform(-max_diff_norm, max_diff_norm, size = 2)
    #             # conceptually: / length followed by * r
    #     purturb_state = true_state + x
    #     purturb_states.append(purturb_state)
    #     tensor_purturb_state = torch.from_numpy(np.ascontiguousarray(purturb_state)).unsqueeze(0).cuda().to(torch.float32)
    #     tensor_purturb_states.append(tensor_purturb_state)
    purturb_states = np.clip((np.random.uniform(-max_diff_norm, max_diff_norm, size = (sample_time,2)) + true_state), 0 ,10)
    tensor_purturb_states = torch.from_numpy(np.ascontiguousarray(purturb_states)).unsqueeze(1).cuda().to(torch.float32)
    actions = get_defender_policy_batch(model, tensor_purturb_states)
    for action, purturb_state in zip(actions,purturb_states):
        if action not in possible_action:
            possible_action.add(action)
        #print(model.forward(true_state_tensor))
        values[tuple(purturb_state)] = model.forward(tensor_true_state)[0][action].data.cpu()
        if len(possible_action) == model.env.action_space.n:
            break
    att_state = min(values, key = values.get)
    return att_state

def get_defender_policy(model, X, max_diff_norm = 0.1, max_iter= 10):
    values = dict()
    loss_fn = torch.nn.CrossEntropyLoss()
    for action in range(model.env.action_space.n):
        #state_tensor = torch.from_numpy(np.ascontiguousarray(purturb_state)).unsqueeze(0).cuda().to(torch.float32).requires_grad
        #optimizer = optim.Adam(state_tensor, lr= 0.001)
        #for niters in range(max_iter):
        step_size = 0.5 / max_iter
        #y = Variable(torch.tensor(y))
        X_adv = Variable(X.data, requires_grad=True)
        #v_t = model.act(X)
        #v_t = Variable(torch.tensor(v_t))
        for i in range(max_iter):
            v = model.forward(X_adv)
            #loss = loss_fn(v_f, v_t).requires_grad_()
            #print(loss)
            loss = v[0][action].requires_grad_()
            #print(loss)
            #loss = loss_fn(v_t, v_f).requires_grad_()
            #loss = (v_t - v_f).requires_grad_()
            #X_adv.zero_grad()
            model.features.zero_grad()
            loss.backward()
            #torch.autograd.grad(loss, X_adv)
            #print(X_adv.grad)
            eta = step_size * X_adv.grad.data.sign()
            X_adv = Variable(X_adv.data + eta, requires_grad=True)
            # adjust to be within [-epsilon, epsilon]
            eta = torch.clamp(X_adv.data - X.data, -max_diff_norm, max_diff_norm)
            X_adv.data = X.data + eta
        values[action] = model.forward(X_adv)[0][action].data.cpu()
    result = max(values, key = values.get)
    return result

def get_defender_policy_batch(model, X, max_diff_norm = 0.5, max_iter = 10):
    loss_fn = torch.nn.CrossEntropyLoss()
    values_list = []
    for i in range(len(X)):
        values_list.append(dict())
    for action in range(model.env.action_space.n):
        #state_tensor = torch.from_numpy(np.ascontiguousarray(purturb_state)).unsqueeze(0).cuda().to(torch.float32).requires_grad
        #optimizer = optim.Adam(state_tensor, lr= 0.001)
        #for niters in range(max_iter):
        step_size = 0.5 / max_iter
        #y = Variable(torch.tensor(y))
        X_adv = Variable(X.data, requires_grad=True)
        #v_t = model.act(X)
        #v_t = Variable(torch.tensor(v_t))
        for i in range(max_iter):
            v = model.forward(X_adv)
            #loss = loss_fn(v_f, v_t).requires_grad_()
            #print(loss)
            #print(v.shape)
            loss = (v[:,:,action][0]).requires_grad_()
            #print(loss)
            #loss = loss_fn(v_t, v_f).requires_grad_()
            #loss = (v_t - v_f).requires_grad_()
            #X_adv.zero_grad()
            model.features.zero_grad()
            loss.backward()
            #torch.autograd.grad(loss, X_adv)
            #print(X_adv.grad)
            eta = step_size * X_adv.grad.data.sign()
            X_adv = Variable(X_adv.data + eta, requires_grad=True)
            # adjust to be within [-epsilon, epsilon]
            eta = torch.clamp(X_adv.data - X.data, -max_diff_norm, max_diff_norm)
            X_adv.data = X.data + eta
        count = 0
        for adv in X_adv:
            values_list[count][action] = model.forward(adv)[0][action].data.cpu()
            count+=1
    result = []
    for value in values_list:
        result.append(max(value, key = value.get))
    return result


def find_maxmin_with_belief(model, next_states, belief):
    result = []
    for next_state, possible_states in zip(next_states, belief):
        possible_action = set()
        value = dict()
        tensor_states = []
        # for state in possible_states:
        #     state_tensor = torch.tensor(state).cuda()
        #     tensor_states.append(state_tensor)
        # tensor_states = torch.stack(tensor_states)
        tensor_states = torch.from_numpy(np.asarray(possible_states)).cuda()
        #print(model.forward(tensor_states))
        actions = model.forward(tensor_states).max(1)[1].data.cpu().numpy()
        for action in actions:
            #print(actions)
            #action = model.act(state_tensor, 0)[0].cpu().numpy()
            possible_action.add(int(action))
        #print(possible_action)
        for action in possible_action:
            value[action] = model.forward(next_state)[action].data.cpu()
        result.append(np.min(list(value.values())))
    result = np.asarray(result)
    result_tensor = torch.from_numpy(result).cuda()
    return result_tensor

def find_maxmin_with_belief_conservative(model, next_states, belief):
    result = []
    for next_state, possible_states in zip(next_states, belief):
        possible_action = set()
        value = []
        tensor_states = []
        # for state in possible_states:
        #     state_tensor = torch.tensor(state).cuda()
        #     tensor_states.append(state_tensor)
        tensor_states = torch.from_numpy(np.asarray(possible_states)).cuda()
        #tensor_states = torch.stack(tensor_states)
        #print(model.forward(tensor_states).max(1))
        values = model.forward(tensor_states).max(1)[0].data.cpu().numpy()
        # for tensor_state, action in zip(tensor_states, actions):
        #     value.append(model.forward(tensor_state)[action].data.cpu())
        result.append(np.min(values))
    result = np.asarray(result)
    result_tensor = torch.from_numpy(result).cuda()
    return result_tensor

def find_maxmin(model, next_states, max_diff_norm = 0.5, max_iter = 10, sample_time = 100):
    d = model.env.observation_space.shape[0]
    result = []
    for next_state_tensor in next_states:
        possible_action = set()
        values = dict()
        #next_state_tensor = torch.from_numpy(np.ascontiguousarray(next_state)).unsqueeze(0).cuda().to(torch.float32)
        next_state = next_state_tensor.data.cpu().numpy()
        possible_next_states = []
        # for _ in range(sample_time):
        #     # inv_d = 1.0 / d
        #     # gauss = np.random.normal(size=d)
        #     # length = np.linalg.norm(gauss)
        #     # if length == 0.0:
        #     #     x = gauss
        #     # else:
        #     #     r = np.random.uniform(0, max_diff_norm)
        #     #     x = np.multiply(gauss, r / length)
        #             # conceptually: / length followed by * r
        #     x = np.random.uniform(-max_diff_norm, max_diff_norm, size = 2)
        #     purturb_state = next_state + x
        #     possible_next_states.append(torch.from_numpy(np.ascontiguousarray(purturb_state)).unsqueeze(0).cuda().to(torch.float32))
        #tensor_possible_next_states = torch.stack(possible_next_states)
        upper_q, lower_q = network_bounds(model, next_state_tensor, max_diff_norm)
        upper_q = upper_q.unsqueeze(0)
        lower_q = lower_q.unsqueeze(0)
        q_value = model.forward(next_state_tensor)
        q_value = q_value.unsqueeze(0)
        actions_tensor = worst_action_select(q_value, upper_q, lower_q, 'cuda')
        actions = actions_tensor.data.cpu().numpy()[0]
        #actions = get_defender_policy_batch(model, tensor_possible_next_states)

        for action in actions:
            if action not in possible_action:
                possible_action.add(action)
            #print(model.forward(true_state_tensor))
            values[action] = model.forward(next_state_tensor)[action].data.cpu()
            if len(possible_action) == model.env.action_space.n:
                break
            #tensor_purturb_state = torch.from_numpy(np.ascontiguousarray(purturb_state)).unsqueeze(0).cuda().to(torch.float32)
            # action = get_defender_policy(model, tensor_purturb_state, max_diff_norm, max_iter)
            # possible_action.append(action)
            #print(model.forward(true_state_tensor))
            #print(model.forward(next_state_tensor))
            # values[tuple(purturb_state)] = model.forward(next_state_tensor)[action].data.cpu()
            # if len(possible_action) == model.env.action_space.n:
            #     break
        result.append(np.min(list(values.values())))
    result = np.asarray(result)
    result_tensor = torch.from_numpy(result).cuda()
    return result_tensor

class TimeLogger(object):
    def __init__(self):
        self.time_logs = {}

    def log_time(self, time_id, time):
        if time_id not in self.time_logs:
            self.time_logs[time_id] = 0.0
        self.time_logs[time_id] += time

    def __call__(self, time_id, time):
        self.log_time(time_id, time)

    def clear(self):
        self.time_logs = {}

    def print(self):
        print_str = ""
        for t in self.time_logs:
            print_str += "{}={:.4f} ".format(t, self.time_logs[t])
        print(print_str + "\n")

log_time = TimeLogger()


def logits_margin(logits, y):
    comp_logits = logits - torch.zeros_like(logits).scatter(1, torch.unsqueeze(y, 1), 1e10)
    sec_logits, _ = torch.max(comp_logits, dim=1)
    margin = sec_logits - torch.gather(logits, 1, torch.unsqueeze(y, 1)).squeeze(1)
    margin = margin.sum()
    return margin


def compute_td_loss(current_model, target_model, batch_size, replay_buffer, per, use_cpp_buffer, use_async_rb, optimizer, gamma, memory_mgr, robust, our_alg = False, **kwargs):
    t = time.time()
    dtype = kwargs['dtype']
    if per:
        buffer_beta = kwargs['buffer_beta']
        if use_async_rb:
            if not replay_buffer.sample_available():
                replay_buffer.async_sample(batch_size, buffer_beta)
            res = replay_buffer.wait_sample()
            replay_buffer.async_sample(batch_size, buffer_beta)
        else:
            res = replay_buffer.sample(batch_size, buffer_beta)
        if use_cpp_buffer:
            if our_alg:
                state, action, reward, next_state, done, indices, weights, belief = res['obs'], res['act'], res['rew'], res['next_obs'], res['done'], res['indexes'], res['weights'], res['belief']
            else:
                state, action, reward, next_state, done, indices, weights = res['obs'], res['act'], res['rew'], res['next_obs'], res['done'], res['indexes'], res['weights']
        else:
            state, action, reward, next_state, done, weights, indices = res[0], res[1], res[2], res[3], res[4], res[5], res[6]
    else:
        if use_async_rb:
            if replay_buffer.sample_available():
                replay_buffer.async_sample(batch_size)
            res = replay_buffer.wait_sample()
            replay_buffer.async_sample(batch_size)
        else:
            res = replay_buffer.sample(batch_size)
        if use_cpp_buffer:
            state, action, reward, next_state, done, belief = res['obs'], res['act'], res['rew'], res['next_obs'], res['done'], res['belief']
        else:
            state, action, reward, next_state, done = res[0], res[1], res[2], res[3], res[4]
    if use_cpp_buffer and not use_async_rb:
        action = action.transpose()[0].astype(int)
        reward = reward.transpose()[0].astype(int)
        done = done.transpose()[0].astype(int)
    log_time('sample_time', time.time() - t)

    t = time.time()
    if per:
        numpy_weights = weights
    if per:
        state, next_state, action, reward, done, weights = memory_mgr.get_cuda_tensors(state, next_state, action, reward, done, weights)
    else:
        state, next_state, action, reward, done = memory_mgr.get_cuda_tensors(np.asarray(state), np.asarray(next_state), np.asarray(action), np.asarray(reward), np.asarray(done))

    bound_solver = kwargs.get('bound_solver', 'cov')
    optimizer.zero_grad()

    state = state.to(torch.float)
    next_state = next_state.to(torch.float)
    # Normalize input pixel to 0-1
    if dtype in UINTS:
        state /= 255
        next_state /= 255
        state_max = 1.0
        state_min = 0.0
    else:
        state_max = float('inf')
        state_min = float('-inf')
    beta = kwargs.get('beta', 0)

    if robust and bound_solver != 'pgd':
        cur_q_logits = current_model(state, method_opt="forward")
        tgt_next_q_logits = target_model(next_state, method_opt="forward")
    else:
        cur_q_logits = current_model(state)
        tgt_next_q_logits = target_model(next_state)
    if robust:
        eps = kwargs['eps']
    cur_q_value = cur_q_logits.gather(1, action.unsqueeze(1)).squeeze(1)

    if our_alg:
        #tgt_next_q_value = find_maxmin(target_model, next_state, 0.5)
        tgt_next_q_value = find_maxmin_with_belief_conservative(target_model, next_state, belief)
        expected_q_value = reward + gamma * tgt_next_q_value*(1-done)
    else:
        tgt_next_q_value = tgt_next_q_logits.max(1)[0]
        expected_q_value = reward + gamma * tgt_next_q_value * (1 - done)
    '''
    # Merge two states into one batch
    state = state.to(torch.float)
    if dtype in UINTS:
        state /= 255
    state_and_next_state = torch.cat((state, next_state), 0)
    logits = current_model(state_and_next_state)
    cur_q_logits = logits[:state.size(0)]
    cur_next_q_logits = logits[state.size(0):]
    tgt_next_q_value  = tgt_next_q_logits.gather(1, torch.max(cur_next_q_logits, 1)[1].unsqueeze(1)).squeeze(1)
    '''

    if kwargs['natural_loss_fn'] == 'huber':
        loss_fn = torch.nn.SmoothL1Loss(reduction='none')
        loss = loss_fn(cur_q_value, expected_q_value.detach())
    else:
        loss  = (cur_q_value - expected_q_value.detach()).pow(2)
    if per:
        loss = loss * weights
        prios = loss + 1e-5
        weights_norm = np.linalg.norm(numpy_weights)

    batch_cur_q_value = torch.mean(cur_q_value)
    batch_exp_q_value = torch.mean(expected_q_value)
    loss = loss.mean()
    td_loss = loss.clone()

    if robust:
#        print(eps)
        if eps < np.finfo(np.float32).tiny:
            reg_loss = torch.zeros(state.size(0))
            if USE_CUDA:
                reg_loss = reg_loss.cuda()
            if bound_solver == 'pgd':
                labels = torch.argmax(cur_q_logits, dim=1).clone().detach()
                adv_margin = ori_margin = logits_margin(current_model.forward(state), labels)
                optimizer.zero_grad()
        else:
            if bound_solver != 'pgd':
                sa = kwargs.get('sa', None)
                pred = cur_q_logits
                labels = torch.argmax(pred, dim=1).clone().detach()
                c = torch.eye(current_model.num_actions).type_as(state)[labels].unsqueeze(1) - torch.eye(current_model.num_actions).type_as(state).unsqueeze(0)
                I = (~(labels.data.unsqueeze(1) == torch.arange(current_model.num_actions).type_as(labels.data).unsqueeze(0)))
                c = (c[I].view(state.size(0), current_model.num_actions-1, current_model.num_actions))
                sa_labels = sa[labels]
                lb_s = torch.zeros(state.size(0), current_model.num_actions)
                if USE_CUDA:
                    labels = labels.cuda()
                    c = c.cuda()
                    sa_labels = sa_labels.cuda()
                    lb_s = lb_s.cuda()
                env_id = kwargs.get('env_id','')
                if env_id == 'Acrobot-v1':
                    eps_v = get_acrobot_eps(eps)
                    if USE_CUDA:
                        eps_v = eps_v.cuda()
                else:
                    eps_v = eps
                state_ub = torch.clamp(state + eps_v, max=state_max)
                state_lb = torch.clamp(state - eps_v, min=state_min)

                lb = get_logits_lower_bound(current_model, state, state_ub, state_lb, eps_v, c, beta)

                hinge = kwargs.get('hinge', False)
                if hinge:
                   reg_loss, _ = torch.min(lb, dim=1)
                   hinge_c = kwargs.get('hinge_c', 1)
                   reg_loss = torch.clamp(reg_loss, max=hinge_c)
                   reg_loss = - reg_loss
                else:
                    lb = lb_s.scatter(1, sa_labels, lb)
                    reg_loss = CrossEntropyLoss()(-lb, labels)
            else:
                labels = torch.argmax(cur_q_logits, dim=1).clone().detach()
                hinge_c = kwargs.get('hinge_c', 1)
                adv_state = attack(current_model, state, kwargs['attack_config'], logits_margin)
                # print("ori_state", state)
                # print("adv_state", adv_state)
                optimizer.zero_grad()
                adv_margin = logits_margin(current_model.forward(adv_state), labels)
                ori_margin = logits_margin(current_model.forward(state), labels)
                reg_loss = torch.clamp(adv_margin, min=-hinge_c)

        if per:
            reg_loss = reg_loss * weights
        reg_loss = reg_loss.mean()
        kappa = kwargs['kappa']
        loss += kappa * reg_loss

    loss.backward()

    # Gradient clipping.
    grad_norm = 0.0
    max_norm = kwargs['grad_clip']
    if max_norm > 0:
        parameters = current_model.parameters()
        for p in parameters:
            grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = np.sqrt(grad_norm)
        clip_coef = max_norm / (grad_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)

    # update weights
    optimizer.step()

    nn_time = time.time() - t
    log_time('nn_time', time.time() - t)
    t = time.time()
    if per:
        replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    log_time('reweight_time', time.time() - t)

    res = (loss, grad_norm, weights_norm, td_loss, batch_cur_q_value, batch_exp_q_value)
    if robust:
        if bound_solver == 'pgd':
            res += (ori_margin, adv_margin)
        res += (reg_loss,)
    return res


def mini_test(model, config, logger, dtype, num_episodes=10, max_frames_per_episode=30000):
    logger.log('start mini test')
    training_config = config['training_config']
    env_params = training_config['env_params']
    env_params['clip_rewards'] = False
    env_params['episode_life'] = False
    env_id = config['env_id']

    if 'NoFrameskip' not in env_id:
        env = make_atari_cart(env_id)
    else:
        env = make_atari(env_id)
        env = wrap_deepmind(env, **env_params)
        env = wrap_pytorch(env)
    state = env.reset()
    all_rewards = []
    episode_reward = 0

    seed = random.randint(0, sys.maxsize)
    logger.log('reseting env with seed', seed)
    env.seed(seed)
    state = env.reset()

    episode_idx = 1
    this_episode_frame = 1
    for frame_idx in range(1, num_episodes * max_frames_per_episode + 1):
        state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).cuda().to(torch.float32)
        if dtype in UINTS:
            state_tensor /= 255
        action = model.act(state_tensor)[0]
        next_state, reward, done, _ = env.step(action)

        # logger.log(action)
        state = next_state
        episode_reward += reward
        if this_episode_frame == max_frames_per_episode:
            logger.log('maximum number of frames reached in this episode, reset environment!')
            done = True

        if done:
            logger.log('reseting env with seed', seed)
            state = env.reset()
            all_rewards.append(episode_reward)
            logger.log('episode {}/{} reward: {:6g}'.format(episode_idx, num_episodes, all_rewards[-1]))
            episode_reward = 0
            this_episode_frame = 1
            episode_idx += 1
            if episode_idx > num_episodes:
                break
        else:
            this_episode_frame += 1
    return np.mean(all_rewards)


def main(args):
    config = load_config(args)
    prefix = config['env_id']
    training_config = config['training_config']
    if config['name_suffix']:
        prefix += config['name_suffix']
    if config['path_prefix']:
       prefix = os.path.join(config['path_prefix'], prefix)
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    train_log = os.path.join(prefix, 'train.log')
    logger = Logger(open(train_log, "w"))
    logger.log('Command line:', " ".join(sys.argv[:]))
    logger.log(args)
    logger.log('/n')
    logger.log(config)

    env_params = training_config['env_params']
    env_id = config['env_id']
    # if "NoFrameskip" not in env_id:
    #     env = make_atari_cart(env_id)
    # else:
    #     env = make_atari(env_id)
    #     env = wrap_deepmind(env, **env_params)
    #     env = wrap_pytorch(env)
    #env = GridWorld()

    #env = gym.make("LunarLander-v2")

    env = Gridmaze()

    seed = training_config['seed']
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    state = env.reset()
    dtype = state.dtype
    logger.log("env_shape: {}, num of actions: {}".format(env.observation_space.shape, env.action_space.n))
    if "NoFrameskip" in env_id:
        logger.log('action meaning:', env.unwrapped.get_action_meanings()[:env.action_space.n])

    robust = training_config.get('robust', False)
    adv_train = training_config.get('adv_train', False)
    bound_solver = training_config.get('bound_solver', 'cov')
    attack_config = {}
    if adv_train or bound_solver == 'pgd':
        test_config = config['test_config']
        attack_config = training_config["attack_config"]
        adv_ratio = training_config.get('adv_ratio', 1)
        if adv_train:
            logger.log('using adversarial examples for training, adv ratio:', adv_ratio)
        else:
            logger.log('using pgd regularization training')
    if robust or adv_train:
        schedule_start = training_config['schedule_start']
        schedule_length = training_config['schedule_length']
        starting_epsilon=  training_config['start_epsilon']
        end_epsilon = training_config['epsilon']
        epsilon_scheduler = EpsilonScheduler(training_config.get("schedule_type", "linear"), schedule_start, schedule_start+schedule_length-1, starting_epsilon, end_epsilon, 1)
        max_eps = end_epsilon

    model_width = training_config['model_width']
    robust_model = robust and bound_solver != 'pgd'
    dueling = training_config.get('dueling', True)

    current_model = model_setup(env_id, env, robust_model, logger, USE_CUDA, dueling, model_width)
    target_model = model_setup(env_id, env, robust_model, logger, USE_CUDA, dueling, model_width)

    # load_path = training_config["load_model_path"]
    # if  load_path != "" and os.path.exists(load_path):
    #     load_frame = int(re.findall('^.*frame_([0-9]+).pth$',load_path)[0])
    #     logger.log('\ntrain from model {}, current frame index is {}\n'.format(load_path, load_frame))
    #     current_model.features.load_state_dict(torch.load(load_path))
    #     target_model.features.load_state_dict(torch.load(load_path))
    # else:
    #     logger.log('\ntrain from scratch')
    #     load_frame = 1
    #
    # lr = training_config['lr']
    # grad_clip = training_config['grad_clip']
    # natural_loss_fn = training_config['natural_loss_fn']
    # optimizer = optim.Adam(current_model.parameters(), lr=lr, eps=training_config['adam_eps'])
    # # Do not evaluate gradient for target model.
    # for param in target_model.features.parameters():
    #     param.requires_grad = False
    #
    # buffer_config = training_config['buffer_params']
    # replay_initial = buffer_config['replay_initial']
    # buffer_capacity = buffer_config['buffer_capacity']
    # use_cpp_buffer = training_config["cpprb"]
    # use_async_rb = training_config['use_async_rb']
    # num_frames = training_config['num_frames']
    # batch_size = training_config['batch_size']
    # gamma = training_config['gamma']
    #
    # if use_cpp_buffer:
    #     logger.log('using cpp replay buffer')
    #     if use_async_rb:
    #         replay_buffer_ctor = AsyncReplayBuffer(initial_state=state, batch_size=batch_size)
    #     else:
    #         replay_buffer_ctor = cpprb.PrioritizedReplayBuffer
    # else:
    #     logger.log('using python replay buffer')
    # per = training_config['per']
    #
    # if per:
    #     logger.log('using prioritized experience replay.')
    #     alpha = buffer_config['alpha']
    #     buffer_beta_start = buffer_config['buffer_beta_start']
    #     buffer_beta_frames = buffer_config.get('buffer_beta_frames', -1)
    #     if buffer_beta_frames < replay_initial:
    #         buffer_beta_frames = num_frames - replay_initial
    #         logger.log('beffer_beta_frames reset to ', buffer_beta_frames)
    #     buffer_beta_scheduler = BufferBetaScheduler(buffer_beta_start, buffer_beta_frames, start_frame=replay_initial)
    #     if use_cpp_buffer:
    #         replay_buffer = replay_buffer_ctor(size=buffer_capacity,
    #                 # env_dict={"obs": {"shape": state.shape, "dtype": np.uint8},
    #                 env_dict={"obs": {"shape": state.shape, "dtype": dtype},
    #                     "act": {"shape": 1, "dtype": np.uint8},
    #                     "rew": {},
    #                     # "next_obs": {"shape": state.shape, "dtype": np.uint8},
    #                     "next_obs": {"shape": state.shape, "dtype": dtype},
    #                     "done": {}}, alpha=alpha, eps = 0.0)  # We add eps manually in training loop
    #     else:
    #         replay_buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=alpha)
    #
    # else:
    #     logger.log('using regular replay.')
    #     if use_cpp_buffer:
    #         replay_buffer =cpprb.ReplayBuffer(buffer_capacity,
    #                 # {"obs": {"shape": state.shape, "dtype": np.uint8},
    #                 {"obs": {"shape": state.shape, "dtype": dtype},
    #                     "act": {"shape": 1, "dtype": np.uint8},
    #                     "rew": {},
    #                     # "next_obs": {"shape": state.shape, "dtype": np.uint8},
    #                     "next_obs": {"shape": state.shape, "dtype": dtype},
    #                     "done": {}})
    #     else:
    #         replay_buffer = ReplayBuffer(buffer_capacity)
    #
    # update_target(current_model, target_model)
    #
    # act_epsilon_start = training_config['act_epsilon_start']
    # act_epsilon_final = training_config['act_epsilon_final']
    # act_epsilon_decay = training_config['act_epsilon_decay']
    # act_epsilon_method = training_config['act_epsilon_method']
    # if training_config.get('act_epsilon_decay_zero', True):
    #     decay_zero = num_frames
    # else:
    #     decay_zero = None
    # act_epsilon_scheduler = ActEpsilonScheduler(act_epsilon_start, act_epsilon_final, act_epsilon_decay, method=act_epsilon_method, start_frame=replay_initial, decay_zero=decay_zero)
    #
    # # Use optimized cuda memory management
    # memory_mgr = CudaTensorManager(state.shape, batch_size, per, USE_CUDA, dtype=dtype)
    #
    # losses = []
    # td_losses = []
    # batch_cur_q = []
    # batch_exp_q = []
    #
    # sa = None
    # kappa = None
    # hinge = False
    # if robust:
    #     logger.log('using convex relaxation certified classification loss as a regularization!')
    #     kappa = training_config['kappa']
    #     reg_losses = []
    #     sa = np.zeros((current_model.num_actions, current_model.num_actions - 1), dtype = np.int32)
    #     for i in range(sa.shape[0]):
    #         for j in range(sa.shape[1]):
    #             if j < i:
    #                 sa[i][j] = j
    #             else:
    #                 sa[i][j] = j + 1
    #     sa = torch.LongTensor(sa)
    #     hinge = training_config.get('hinge', False)
    #     logger.log('using hinge loss (default is cross entropy): ', hinge)
    #
    #
    # if training_config['use_async_env']:
    #     # Create an environment in a separate process, run asychronously
    #     async_env = AsyncEnv(env_id, result_path=prefix, draw=training_config['show_game'], record=training_config['record_game'], env_params=env_params, seed=seed)
    #
    # # initialize parameters in logging
    # all_rewards = []
    # episode_reward = 0
    # act_epsilon = np.nan
    # grad_norm = np.nan
    # weights_norm = np.nan
    # best_test_reward = -float('inf')
    # buffer_stored_size = 0
    # if adv_train:
    #     attack_count = 0
    #     suc_count = 0
    # if robust and bound_solver == 'pgd':
    #     ori_margin, adv_margin = np.nan, np.nan
    #
    # start_time = time.time()
    # period_start_time = time.time()
    #
    # # Main Loop
    # for frame_idx in range(load_frame, num_frames + 1):
    #     # Step 1: get current action
    #     frame_start = time.time()
    #     t = time.time()
    #
    #     eps = 0
    #     if adv_train or robust:
    #         eps = epsilon_scheduler.get_eps(frame_idx, 0)
    #
    #     act_epsilon = act_epsilon_scheduler.get(frame_idx)
    #     if adv_train and eps != np.nan and eps >= np.finfo(np.float32).tiny:
    #         ori_state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).cuda().to(torch.float32)
    #         if dtype in UINTS:
    #             ori_state_tensor /= 255
    #         attack_config['params']['epsilon'] = eps
    #         if random.random() < adv_ratio:
    #             attack_count += 1
    #             state_tensor = attack(current_model, ori_state_tensor, attack_config)
    #             if current_model.act(state_tensor)[0] != current_model.act(ori_state_tensor)[0]:
    #                 suc_count += 1
    #         else:
    #             state_tensor = ori_state_tensor
    #         action = current_model.act(state_tensor, act_epsilon)[0]
    #     else:
    #         with torch.no_grad():
    #             state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).cuda().to(torch.float32)
    #             if dtype in UINTS:
    #                 state_tensor /= 255
    #             ori_state_tensor = torch.clone(state_tensor)
    #             action = current_model.act(state_tensor, act_epsilon)[0]
    #     # upper_q, lower_q = network_bounds(current_model, state_tensor, 0.5)
    #     # q_value = current_model.forward(state_tensor)
    #     # actions_tensor = worst_action_select(q_value, upper_q, lower_q, 'cuda')
    #     # actions = actions_tensor.data.cpu().numpy()
    #     # print(actions[0])
    #
    #     # torch.cuda.synchronize()
    #     log_time('act_time', time.time() - t)
    #
    #     # Step 2: run environment
    #     t = time.time()
    #     if training_config['use_async_env']:
    #         async_env.async_step(action)
    #     else:
    #         next_state, reward, done, _ = env.step(action)
    #     log_time('env_time', time.time() - t)
    #
    #
    #     # Step 3: save to buffer
    #     # For asynchronous env, defer saving
    #     if not training_config['use_async_env']:
    #         t = time.time()
    #         if use_cpp_buffer:
    #             replay_buffer.add(obs=state, act=action, rew=reward, next_obs=next_state, done=done)
    #         else:
    #             replay_buffer.push(state, action, reward, next_state, done)
    #         log_time('save_time', time.time() - t)
    #
    #     if use_cpp_buffer:
    #         buffer_stored_size = replay_buffer.get_stored_size()
    #     else:
    #         buffer_stored_size = len(replay_buffer)
    #
    #     beta = np.nan
    #     buffer_beta = np.nan
    #     t = time.time()
    #
    #     if buffer_stored_size > replay_initial:
    #         if training_config['per']:
    #             buffer_beta = buffer_beta_scheduler.get(frame_idx)
    #         if robust:
    #             convex_final_beta = training_config['convex_final_beta']
    #             convex_start_beta = training_config['convex_start_beta']
    #             beta = (max_eps - eps * (1.0 - convex_final_beta)) / max_eps * convex_start_beta
    #
    #         res = compute_td_loss(current_model, target_model, batch_size, replay_buffer, per, use_cpp_buffer, use_async_rb, optimizer, gamma, memory_mgr, robust, buffer_beta=buffer_beta, grad_clip=grad_clip, natural_loss_fn=natural_loss_fn, eps=eps, beta=beta, sa=sa, kappa=kappa, dtype=dtype, hinge=hinge, hinge_c=training_config.get('hinge_c', 1), env_id=env_id, bound_solver=bound_solver, attack_config=attack_config)
    #         loss, grad_norm, weights_norm, td_loss, batch_cur_q_value, batch_exp_q_value = res[0], res[1], res[2], res[3], res[4], res[5]
    #         if robust:
    #             reg_loss = res[-1]
    #             reg_losses.append(reg_loss.data.item())
    #             if bound_solver == 'pgd':
    #                 ori_margin, adv_margin = res[-3].data.item(), res[-2].data.item()
    #
    #         losses.append(loss.data.item())
    #         td_losses.append(td_loss.data.item())
    #         batch_cur_q.append(batch_cur_q_value.data.item())
    #         batch_exp_q.append(batch_exp_q_value.data.item())
    #
    #     log_time('loss_time', time.time() - t)
    #
    #     # Step 2: run environment (async)
    #     t = time.time()
    #     if training_config['use_async_env']:
    #         next_state, reward, done, _ = async_env.wait_step()
    #     log_time('env_time', time.time() - t)
    #
    #     # Step 3: save to buffer (async)
    #     if training_config['use_async_env']:
    #         t = time.time()
    #         if use_cpp_buffer:
    #             replay_buffer.add(obs=state, act=action, rew=reward, next_obs=next_state, done=done)
    #         else:
    #             replay_buffer.push(state, action, reward, next_state, done)
    #         log_time('save_time', time.time() - t)
    #
    #     # Update states and reward
    #     t = time.time()
    #     state = next_state
    #     episode_reward += reward
    #     if done:
    #         if training_config['use_async_env']:
    #             state = async_env.reset()
    #         else:
    #             state = env.reset()
    #         all_rewards.append(episode_reward)
    #         episode_reward = 0
    #     log_time('env_time', time.time() - t)
    #
    #     # All kinds of result logging
    #     if frame_idx % training_config['print_frame'] == 0 or frame_idx==num_frames or (robust and abs(frame_idx-schedule_start) < 5) or abs(buffer_stored_size-replay_initial) < 5:
    #         logger.log('\nframe {}/{}, learning rate: {:.6g}, buffer beta: {:.6g}, action epsilon: {:.6g}'.format(frame_idx, num_frames, lr, buffer_beta, act_epsilon))
    #         logger.log('total time: {:.2f}, epoch time: {:.4f}, speed: {:.2f} frames/sec, last total loss: {:.6g}, avg total loss: {:.6g}, grad norm: {:.6g}, weights_norm: {:.6g}, latest episode reward: {:.6g}, avg 10 episode reward: {:.6g}'.format(
    #             time.time() - start_time,
    #             time.time() - period_start_time,
    #             training_config['print_frame'] / (time.time() - period_start_time),
    #             losses[-1] if losses else np.nan,
    #             np.average(losses[:-training_config['print_frame']-1:-1]) if losses else np.nan,
    #             grad_norm, weights_norm,
    #             all_rewards[-1] if all_rewards else np.nan,
    #             np.average(all_rewards[:-11:-1]) if all_rewards else np.nan))
    #         logger.log('last td loss: {:.6g}, avg td loss: {:.6g}'.format(
    #                 td_losses[-1] if td_losses else np.nan,
    #                 np.average(td_losses[:-training_config['print_frame']-1:-1]) if td_losses else np.nan))
    #         logger.log('last batch cur q: {:.6g}, avg batch cur q: {:.6g}'.format(
    #                 batch_cur_q[-1] if batch_cur_q else np.nan,
    #                 np.average(batch_cur_q[:-training_config['print_frame']-1:-1]) if batch_cur_q else np.nan))
    #         logger.log('last batch exp q: {:.6g}, avg batch exp q: {:.6g}'.format(
    #                 batch_exp_q[-1] if batch_exp_q else np.nan,
    #                 np.average(batch_exp_q[:-training_config['print_frame']-1:-1]) if batch_exp_q else np.nan))
    #         if robust:
    #             logger.log('current input epsilon: {:.6g}'.format(eps))
    #             if bound_solver == 'pgd':
    #                 logger.log('last logit margin: ori: {:.6g}, adv: {:.6g}'.format(ori_margin, adv_margin))
    #             else:
    #                 logger.log('current bound beta: {:.6g}'.format(beta))
    #             logger.log('last cert reg loss: {:.6g}, avg cert reg loss: {:.6g}'.format(
    #                 reg_losses[-1] if reg_losses else np.nan,
    #                 np.average(reg_losses[:-training_config['print_frame']-1:-1]) if reg_losses else np.nan))
    #             logger.log('current kappa: {:.6g}'.format(kappa))
    #         if adv_train:
    #             logger.log('current attack epsilon (same as input epsilon): {:.6g}'.format(eps))
    #             diff = ori_state_tensor - state_tensor
    #             diff = np.abs(diff.data.cpu().numpy())
    #             logger.log('current Linf distortion: {:.6g}'.format(np.max(diff)))
    #             logger.log('this batch attacked: {}, success: {}, attack success rate: {:.6g}'.format(attack_count, suc_count, suc_count*1.0/attack_count if attack_count>0 else np.nan))
    #             attack_count = 0
    #             suc_count = 0
    #             logger.log('attack stats reseted.')
    #
    #         period_start_time = time.time()
    #         log_time.print()
    #         log_time.clear()
    #
    #     if frame_idx % training_config['save_frame'] == 0 or frame_idx==num_frames:
    #         plot(frame_idx, all_rewards, losses, prefix)
    #         torch.save(current_model.features.state_dict(), '{}/frame_{}.pth'.format(prefix, frame_idx))
    #
    #     if frame_idx % training_config['update_target_frame'] == 0:
    #         update_target(current_model, target_model)
    #
    #     # if frame_idx % training_config.get('mini_test', 100000) == 0 and ((robust and beta == 0) or (not robust and frame_idx * 1.0 / num_frames >= 0.8)):
    #     #     test_reward = mini_test(current_model, config, logger, dtype)
    #     #     logger.log('this test avg reward: {:6g}'.format(test_reward))
    #     #     if test_reward >= best_test_reward:
    #     #         best_test_reward = test_reward
    #     #         logger.log('new best reward {:6g} achieved, update checkpoint'.format(test_reward))
    #     #         torch.save(current_model.features.state_dict(), '{}/best_frame_{}.pth'.format(prefix, frame_idx))
    #
    #     log_time.log_time('total', time.time() - frame_start)

    our_alg = True
    if our_alg:
        state = env.reset()
        dtype = state.dtype

        model_width = training_config['model_width']
        robust_model = robust and bound_solver != 'pgd'
        dueling = training_config.get('dueling', True)

        ori_current_model = copy.deepcopy(current_model)
        ori_target_model = copy.deepcopy(target_model)
        current_model = current_model
        target_model = target_model



        #load_path = training_config["load_model_path"]
        load_path = 'pretrained/grid.pth'
        if  load_path != "" and os.path.exists(load_path):
            load_frame = int(re.findall('^.*frame_([0-9]+).pth$',load_path)[0])
            logger.log('\ntrain from model {}, current frame index is {}\n'.format(load_path, load_frame))
            current_model.features.load_state_dict(torch.load(load_path))
            target_model.features.load_state_dict(torch.load(load_path))
        else:
            logger.log('\ntrain from scratch')
            load_frame = 1

        lr = training_config['lr']
        grad_clip = training_config['grad_clip']
        natural_loss_fn = training_config['natural_loss_fn']
        optimizer = optim.Adam(current_model.parameters(), lr=lr, eps=training_config['adam_eps'])
        # Do not evaluate gradient for target model.
        for param in target_model.features.parameters():
            param.requires_grad = False

        buffer_config = training_config['buffer_params']
        replay_initial = buffer_config['replay_initial']
        buffer_capacity = buffer_config['buffer_capacity']
        use_cpp_buffer = training_config["cpprb"]
        #use_cpp_buffer = False
        #use_async_rb = training_config['use_async_rb']
        use_async_rb = False
        num_frames = 6000000
        batch_size = training_config['batch_size']
        gamma = training_config['gamma']

        if use_cpp_buffer:
            logger.log('using cpp replay buffer')
            if use_async_rb:
                replay_buffer_ctor = AsyncReplayBuffer(initial_state=state, batch_size=batch_size)
            else:
                replay_buffer_ctor = cpprb.PrioritizedReplayBuffer
        else:
            logger.log('using python replay buffer')
        per = training_config['per']

        if per:
            logger.log('using prioritized experience replay.')
            alpha = buffer_config['alpha']
            buffer_beta_start = buffer_config['buffer_beta_start']
            buffer_beta_frames = buffer_config.get('buffer_beta_frames', -1)
            if buffer_beta_frames < replay_initial:
                buffer_beta_frames = num_frames - replay_initial
                logger.log('beffer_beta_frames reset to ', buffer_beta_frames)
            buffer_beta_scheduler = BufferBetaScheduler(buffer_beta_start, buffer_beta_frames, start_frame=replay_initial)
            if use_cpp_buffer:
                replay_buffer = replay_buffer_ctor(size=buffer_capacity,
                        # env_dict={"obs": {"shape": state.shape, "dtype": np.uint8},
                        env_dict={"obs": {"shape": state.shape, "dtype": dtype},
                            "act": {"shape": 1, "dtype": np.uint8},
                            "rew": {},
                            # "next_obs": {"shape": state.shape, "dtype": np.uint8},
                            "next_obs": {"shape": state.shape, "dtype": dtype},
                            "belief":{"shape": (30,2)},
                            "done": {}}, alpha=alpha, eps = 0.0)  # We add eps manually in training loop
            else:
                replay_buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=alpha)

        else:
            logger.log('using regular replay.')
            if use_cpp_buffer:
                replay_buffer =cpprb.ReplayBuffer(buffer_capacity,
                        # {"obs": {"shape": state.shape, "dtype": np.uint8},
                        {"obs": {"shape": state.shape, "dtype": dtype},
                            "act": {"shape": 1, "dtype": np.uint8},
                            "rew": {},
                            # "next_obs": {"shape": state.shape, "dtype": np.uint8},
                            "next_obs": {"shape": state.shape, "dtype": dtype},
                            "done": {}})
            else:
                replay_buffer = ReplayBuffer(buffer_capacity)

        update_target(current_model, target_model)

        act_epsilon_start = training_config['act_epsilon_start']
        act_epsilon_final = training_config['act_epsilon_final']
        act_epsilon_decay = training_config['act_epsilon_decay']
        act_epsilon_method = training_config['act_epsilon_method']
        if training_config.get('act_epsilon_decay_zero', True):
            decay_zero = num_frames
        else:
            decay_zero = None
        act_epsilon_scheduler = ActEpsilonScheduler(act_epsilon_start, act_epsilon_final, act_epsilon_decay, method=act_epsilon_method, start_frame=replay_initial, decay_zero=decay_zero)

        # Use optimized cuda memory management
        memory_mgr = CudaTensorManager(state.shape, batch_size, per, USE_CUDA, dtype=dtype)

        losses = []
        td_losses = []
        batch_cur_q = []
        batch_exp_q = []

        sa = None
        kappa = None
        hinge = False
        if robust:
            logger.log('using convex relaxation certified classification loss as a regularization!')
            kappa = training_config['kappa']
            reg_losses = []
            sa = np.zeros((current_model.num_actions, current_model.num_actions - 1), dtype = np.int32)
            for i in range(sa.shape[0]):
                for j in range(sa.shape[1]):
                    if j < i:
                        sa[i][j] = j
                    else:
                        sa[i][j] = j + 1
            sa = torch.LongTensor(sa)
            hinge = training_config.get('hinge', False)
            logger.log('using hinge loss (default is cross entropy): ', hinge)

        training_config['use_async_env'] = False
        if training_config['use_async_env']:
            # Create an environment in a separate process, run asychronously
            async_env = AsyncEnv(env_id, result_path=prefix, draw=training_config['show_game'], record=training_config['record_game'], env_params=env_params, seed=seed)

        # initialize parameters in logging
        all_rewards = []
        episode_reward = 0
        act_epsilon = np.nan
        grad_norm = np.nan
        weights_norm = np.nan
        best_test_reward = -float('inf')
        buffer_stored_size = 0
        if adv_train:
            attack_count = 0
            suc_count = 0
        if robust and bound_solver == 'pgd':
            ori_margin, adv_margin = np.nan, np.nan

        start_time = time.time()
        period_start_time = time.time()

        # Main Loop
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
        belief_model = get_model(belief_args).cuda()
        model_checkpoint, _ = get_checkpoint(belief_args)
        # print(model_checkpoint)
        belief_model.load_state_dict(model_checkpoint)
        belief = []
        for frame_idx in range(1, num_frames + 1):
            # Step 1: get current action
            frame_start = time.time()
            t = time.time()
            if frame_idx % 100 == 0:
                print(frame_idx)

            eps = 0
            if adv_train or robust:
                eps = epsilon_scheduler.get_eps(frame_idx, 0)

            act_epsilon = act_epsilon_scheduler.get(frame_idx)
            if adv_train and eps != np.nan and eps >= np.finfo(np.float32).tiny:
                ori_state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).cuda().to(torch.float32)
                if dtype in UINTS:
                    ori_state_tensor /= 255
                attack_config['params']['epsilon'] = eps
                if random.random() < adv_ratio:
                    attack_count += 1
                    state_tensor = attack(current_model, ori_state_tensor, attack_config)
                    if current_model.act(state_tensor)[0] != current_model.act(ori_state_tensor)[0]:
                        suc_count += 1
                else:
                    state_tensor = ori_state_tensor
                action = current_model.act(state_tensor, act_epsilon)[0]
            else:
                #with torch.no_grad():
                state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).cuda().to(torch.float32)
                #att_state = get_att_state(current_model, state)
                att_state_tensor = pgd(current_model, state_tensor, current_model.act(state_tensor), env_id = "Grid")
                if dtype in UINTS:
                    state_tensor /= 255
                ori_state_tensor = torch.clone(state_tensor)
                #att_state_tensor = torch.from_numpy(np.ascontiguousarray(att_state)).unsqueeze(0).cuda().to(torch.float32)
                att_state = att_state_tensor.cpu().numpy()[0]
                if len(belief)==0:
                    action = current_model.act(att_state_tensor, act_epsilon)[0]
                else:
                    eps = linear_schedule_exp(0.001, 0.05, 1000000, frame_idx)
                    action = get_action_withbelief(env, current_model, belief, eps)
                    #print(action)

                #record history trajectory

                obs.append(att_state)
                h = env.robbie.h / 360 * 2 * np.pi
                temp_state = copy.deepcopy(state)
                temp_state = list(temp_state)
                temp_state.append(h)
                temp_state = np.asarray(temp_state)
                pos.append(temp_state)
                actions.append([action])
                obs_tensor = torch.Tensor(np.asarray(obs)).unsqueeze(0).cuda()
                pos_tensor = torch.Tensor(np.asarray(pos)).unsqueeze(0).cuda()
                actions_tensor = torch.Tensor(np.asarray(actions)).unsqueeze(0).cuda()

                belief = get_belief(belief_model, belief_args, env_map, obs_tensor, pos_tensor, actions_tensor)
                #belief = random_belief(att_state)
                #print(belief.shape)
            # torch.cuda.synchronize()
            log_time('act_time', time.time() - t)

            # Step 2: run environment
            t = time.time()
            if training_config['use_async_env']:
                async_env.async_step(action)
            else:
                next_state, reward, done, _ = env.step(action)
                #print(env.robbie.x, env.robbie.y)
            log_time('env_time', time.time() - t)


            # Step 3: save to buffer
            # For asynchronous env, defer saving
            if not training_config['use_async_env']:
                t = time.time()
                if use_cpp_buffer:
                    replay_buffer.add(obs=state, act=action, rew=reward, next_obs=next_state, belief=belief, done=done)
                else:
                    replay_buffer.push(state, action, reward, next_state, done)
                log_time('save_time', time.time() - t)

            if use_cpp_buffer:
                buffer_stored_size = replay_buffer.get_stored_size()
            else:
                buffer_stored_size = len(replay_buffer)

            beta = np.nan
            buffer_beta = np.nan
            t = time.time()

            if buffer_stored_size > replay_initial:
                if training_config['per']:
                    buffer_beta = buffer_beta_scheduler.get(frame_idx)
                if robust:
                    convex_final_beta = training_config['convex_final_beta']
                    convex_start_beta = training_config['convex_start_beta']
                    beta = (max_eps - eps * (1.0 - convex_final_beta)) / max_eps * convex_start_beta

                res = compute_td_loss(current_model, target_model, batch_size, replay_buffer, per, use_cpp_buffer, use_async_rb, optimizer, gamma, memory_mgr, robust, buffer_beta=buffer_beta, grad_clip=grad_clip, natural_loss_fn=natural_loss_fn, eps=eps, beta=beta, sa=sa, kappa=kappa, dtype=dtype, hinge=hinge, hinge_c=training_config.get('hinge_c', 1), env_id=env_id, bound_solver=bound_solver, our_alg= our_alg, attack_config=attack_config)
                loss, grad_norm, weights_norm, td_loss, batch_cur_q_value, batch_exp_q_value = res[0], res[1], res[2], res[3], res[4], res[5]
                if robust:
                    reg_loss = res[-1]
                    reg_losses.append(reg_loss.data.item())
                    if bound_solver == 'pgd':
                        ori_margin, adv_margin = res[-3].data.item(), res[-2].data.item()

                losses.append(loss.data.item())
                td_losses.append(td_loss.data.item())
                batch_cur_q.append(batch_cur_q_value.data.item())
                batch_exp_q.append(batch_exp_q_value.data.item())

            log_time('loss_time', time.time() - t)

            # Step 2: run environment (async)
            t = time.time()
            if training_config['use_async_env']:
                next_state, reward, done, _ = async_env.wait_step()
            log_time('env_time', time.time() - t)

            # Step 3: save to buffer (async)
            if training_config['use_async_env']:
                t = time.time()
                if use_cpp_buffer:
                    replay_buffer.add(obs=state, act=action, rew=reward, next_obs=next_state, belief = belief, done=done)
                else:
                    replay_buffer.push(state, action, reward, next_state, done)
                log_time('save_time', time.time() - t)

            # Update states and reward
            t = time.time()
            state = next_state
            episode_reward += reward
            if done:
                obs = []
                pos = []
                actions = []
                if training_config['use_async_env']:
                    state = async_env.reset()
                else:
                    state = env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
            log_time('env_time', time.time() - t)

            # All kinds of result logging
            if frame_idx % training_config['print_frame'] == 0 or frame_idx==num_frames or (robust and abs(frame_idx-schedule_start) < 5) or abs(buffer_stored_size-replay_initial) < 5:
                logger.log('\nframe {}/{}, learning rate: {:.6g}, buffer beta: {:.6g}, action epsilon: {:.6g}'.format(frame_idx, num_frames, lr, buffer_beta, act_epsilon))
                logger.log('total time: {:.2f}, epoch time: {:.4f}, speed: {:.2f} frames/sec, last total loss: {:.6g}, avg total loss: {:.6g}, grad norm: {:.6g}, weights_norm: {:.6g}, latest episode reward: {:.6g}, avg 10 episode reward: {:.6g}'.format(
                    time.time() - start_time,
                    time.time() - period_start_time,
                    training_config['print_frame'] / (time.time() - period_start_time),
                    losses[-1] if losses else np.nan,
                    np.average(losses[:-training_config['print_frame']-1:-1]) if losses else np.nan,
                    grad_norm, weights_norm,
                    all_rewards[-1] if all_rewards else np.nan,
                    np.average(all_rewards[:-11:-1]) if all_rewards else np.nan))
                logger.log('last td loss: {:.6g}, avg td loss: {:.6g}'.format(
                        td_losses[-1] if td_losses else np.nan,
                        np.average(td_losses[:-training_config['print_frame']-1:-1]) if td_losses else np.nan))
                logger.log('last batch cur q: {:.6g}, avg batch cur q: {:.6g}'.format(
                        batch_cur_q[-1] if batch_cur_q else np.nan,
                        np.average(batch_cur_q[:-training_config['print_frame']-1:-1]) if batch_cur_q else np.nan))
                logger.log('last batch exp q: {:.6g}, avg batch exp q: {:.6g}'.format(
                        batch_exp_q[-1] if batch_exp_q else np.nan,
                        np.average(batch_exp_q[:-training_config['print_frame']-1:-1]) if batch_exp_q else np.nan))
                if robust:
                    logger.log('current input epsilon: {:.6g}'.format(eps))
                    if bound_solver == 'pgd':
                        logger.log('last logit margin: ori: {:.6g}, adv: {:.6g}'.format(ori_margin, adv_margin))
                    else:
                        logger.log('current bound beta: {:.6g}'.format(beta))
                    logger.log('last cert reg loss: {:.6g}, avg cert reg loss: {:.6g}'.format(
                        reg_losses[-1] if reg_losses else np.nan,
                        np.average(reg_losses[:-training_config['print_frame']-1:-1]) if reg_losses else np.nan))
                    logger.log('current kappa: {:.6g}'.format(kappa))
                if adv_train:
                    logger.log('current attack epsilon (same as input epsilon): {:.6g}'.format(eps))
                    diff = ori_state_tensor - state_tensor
                    diff = np.abs(diff.data.cpu().numpy())
                    logger.log('current Linf distortion: {:.6g}'.format(np.max(diff)))
                    logger.log('this batch attacked: {}, success: {}, attack success rate: {:.6g}'.format(attack_count, suc_count, suc_count*1.0/attack_count if attack_count>0 else np.nan))
                    attack_count = 0
                    suc_count = 0
                    logger.log('attack stats reseted.')

                period_start_time = time.time()
                log_time.print()
                log_time.clear()

            if frame_idx % training_config['save_frame'] == 0 or frame_idx==num_frames:
                plot(frame_idx, all_rewards, losses, prefix)
                torch.save(current_model.features.state_dict(), '{}/att_conservative_frame_{}.pth'.format(prefix, frame_idx))

            if frame_idx % training_config['update_target_frame'] == 0:
                update_target(current_model, target_model)

            # if frame_idx % training_config.get('mini_test', 100000) == 0 and ((robust and beta == 0) or (not robust and frame_idx * 1.0 / num_frames >= 0.8)):
            #     test_reward = mini_test(current_model, config, logger, dtype)
            #     logger.log('this test avg reward: {:6g}'.format(test_reward))
            #     if test_reward >= best_test_reward:
            #         best_test_reward = test_reward
            #         logger.log('new best reward {:6g} achieved, update checkpoint'.format(test_reward))
            #         torch.save(current_model.features.state_dict(), '{}/best_frame_{}.pth'.format(prefix, frame_idx))

            #log_time.log_time('total', time.time() - frame_start)



if __name__ == "__main__":
    args = argparser()
    main(args)
