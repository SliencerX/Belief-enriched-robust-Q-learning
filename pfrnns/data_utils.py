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
import numpy as np
import random
import gym
import sys
sys.path.append('..')
sys.path.append("../common")
from autoencoder_atari import *
from atari_utils import *
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch, make_atari_cart
from models import *

class Maze(object):
    def __init__(self, maze):
        self.maze = maze
        self.width = len(maze[0])
        self.height = len(maze)
        self.blocks = []
        self.update_cnt = 0
        self.beacons = []
        for y, line in enumerate(self.maze):
            for x, block in enumerate(line):
                if block:
                    nb_y = self.height - y - 1
                    self.blocks.append((x, nb_y))
                    if block == 2:
                        self.beacons.extend(((x, nb_y), (x + 1, nb_y), (x, nb_y + 1), (x + 1, nb_y + 1)))
        #print(self.beacons)

    def is_in(self, x, y):
        if x < 0 or y < 0 or x > self.width or y > self.height:
            return False
        return True

    def is_free(self, x, y):
        if not self.is_in(x, y):
            return False

        yy = self.height - int(y) - 1
        xx = int(x)
        return self.maze[yy][xx] == 0

    def random_place(self):
        x = random.uniform(0, self.width)
        y = random.uniform(0, self.height)
        return x, y

    def random_free_place(self):
        while True:
            x, y = self.random_place()
            if self.is_free(x, y):
                return x, y

    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def distance_to_beacons(self, x, y, obs_num=5):
        distances = []
        for c_x, c_y in self.beacons:
            d = self.distance(c_x, c_y, x, y)
            distances.append(d)
        return sorted(distances)[:obs_num]


def add_noise(level, *coords):
    return [x + random.uniform(-level, level) for x in coords]


def add_noise_gauss(level, *coords):
    return [x + np.random.normal(scale=level) for x in coords]
    # return [x + random.uniform(-level, level) for x in coords]


def add_little_noise(*coords):
    return add_noise(0.02, *coords)


def add_some_noise(*coords):
    return add_noise(0.1, *coords)


class Point(object):
    def __init__(self, x, y, heading=None, w=1, noisy=False):
        if heading is None:
            heading = random.uniform(0, 360)
        if noisy:
            x, y, heading = add_some_noise(x, y, heading)

        self.x = x
        self.y = y
        self.h = heading
        self.w = w

    def __repr__(self):
        return "(%f, %f, w=%f)" % (self.x, self.y, self.w)

    @property
    def xy(self):
        return self.x, self.y

    @property
    def xyh(self):
        return self.x, self.y, self.h

    @classmethod
    def create_random(cls, count, maze):
        return [cls(*maze.random_free_place()) for _ in range(0, count)]

    def read_sensor(self, maze, obs_num):
        return maze.distance_to_beacons(*self.xy, obs_num)

    def advance_by(self, speed, checker=None, noisy=False):
        h = self.h
        if noisy:
            speed, h = add_little_noise(speed, h)
            h += random.uniform(-3, 3)  # needs more noise to disperse better
        r = math.radians(h)
        dx = math.sin(r) * speed
        dy = math.cos(r) * speed
        if checker is None or checker(self, dx, dy):
            self.move_by(dx, dy)
            return True
        return False

    def move_by(self, x, y):
        self.x += x
        self.y += y


class Robot(Point):
    def __init__(self, maze, speed=0.2):
        super(Robot, self).__init__(*maze.random_free_place(), heading=90)
        self.chose_random_direction()
        self.step_count = 0
        self.speed = speed

    def chose_random_direction(self):
        heading = random.uniform(0, 360)
        self.h = heading

    def read_sensor(self, maze, obs_num):
        obs = super(Robot, self).read_sensor(maze, obs_num)
        level = 0.1
        return [x + random.uniform(-level, level) for x in obs]

    def move(self, maze):
        """
        Move the robot. Note that the movement is stochastic too.
        """
        while True:
            self.step_count += 1
            if self.advance_by(self.speed, noisy=False,
                               checker=lambda r, dx, dy: maze.is_free(r.x + dx, r.y + dy)):
                break
            # Bumped into something or too long in same direction,
            # chose random new direction
            self.chose_random_direction()
    def move_one(self,maze):
        self.step_count += 1
        self.advance_by(self.speed, noisy = False, checker=lambda r, dx, dy: maze.is_free(r.x + dx, r.y + dy))


# def gen_traj(traj_len=100, obs_num=5):
#     maze_data = np.loadtxt('maze.csv', delimiter=',')
#
#     world = Maze(maze_data)
#
#     speed = 0.2
#     robbie = Robot(world, speed=speed)
#     traj_ret = []
#
#     for _ in range(traj_len):
#         #r_d = robbie.read_sensor(world, obs_num)
#         step_data = [robbie.x, robbie.y, robbie.h]
#         r_d = [x + random.uniform(-0.1, 0.1) for x in step_data[:2]]
#         r_d.append(robbie.h)
#         old_heading = robbie.h
#         old_x = robbie.x
#         old_y = robbie.y
#         robbie.move(world)
#         d_h = robbie.h - old_heading
#         d_x = robbie.x - old_x
#         d_y = robbie.y - old_y
#
#         action = [d_x, d_y, d_h]
#
#         step_data = step_data + action + r_d
#
#         traj_ret.append(step_data)
#
#     return np.array(traj_ret), maze_data


def gen_traj(traj_len=100, obs_num=5):
    maze_data = np.loadtxt('maze.csv', delimiter=',')

    world = Gridmaze()

    speed = 0.5
    robbie = Robot(world.world, speed=speed)
    traj_ret = []

    for _ in range(traj_len):
        #r_d = robbie.read_sensor(world, obs_num)
        step_data = [robbie.x, robbie.y, robbie.h]
        r_d = [x + random.uniform(-0.5, 0.5) for x in step_data[:2]]
        r_d.append(robbie.h)
        old_heading = robbie.h
        old_x = robbie.x
        old_y = robbie.y
        action = np.random.randint(0,7)
        robbie.h = world.direction[action]
        robbie.move_one(world.world)
        d_h = robbie.h - old_heading
        d_x = robbie.x - old_x
        d_y = robbie.y - old_y

        # action = [d_x, d_y, d_h]

        step_data = step_data + [action] + r_d

        traj_ret.append(step_data)

    return np.array(traj_ret), maze_data

def gen_data(num_trajs, traj_len=50, obs_num=3):
    data = {
        'trajs': []
    }

    from tqdm import tqdm
    for _ in tqdm(range(num_trajs)):
        traj_data, maze = gen_traj(traj_len, obs_num)
        data['trajs'].append(traj_data)

    data['map'] = maze

    return data
# maze_data = np.loadtxt('maze.csv', delimiter=',')
#
# world = Maze(maze_data)
class Gridmaze(gym.Env):
    def __init__(self):
        self.maze_data = np.loadtxt('maze.csv', delimiter=',')
        self.world = Maze(self.maze_data)
        self.speed = 0.5
        self.robbie = Robot(self.world, speed=self.speed)
        self.action_space = spaces.Discrete(8)
        self.direction = [0,45,90,135,180,225,270,315]
        self.observation_space = spaces.Box(low = np.array([0,0]), high = np.array([10,10]), dtype = np.float32)
        self.gold_loc = [9,4]
        self.bomb_loc = [9,5]
        self.count = 0

    def check_loc(self, x,y):
        return [int(x), int(y)]

    def step(self, action):
        self.count += 1
        self.robbie.h = self.direction[action]
        self.robbie.move_one(self.world)
        obs = [self.robbie.x, self.robbie.y]
        loc = self.check_loc(self.robbie.x, self.robbie.y)
        done = False
        if loc == self.gold_loc:
            done = True
            reward = 100
        elif loc == self.bomb_loc:
            done = True
            reward = -100
        else:
            done = False
            reward = -1
        if self.count >= 1000:
            done = True
        return np.array(obs), reward, done, {}

    def reset(self):
        self.robbie = Robot(self.world, speed=self.speed)
        self.count = 0

        return np.asarray([self.robbie.x, self.robbie.y])

def gen_traj_atari(encoder, autoencoder, traj_len=100, obs_num=5):

    env_params = {}
    env_params["crop_shift"] = 10
    env_params["restrict_actions"] = 4
    env_id = "PongNoFrameskip-v4"
    # env = gym.make(env_id)
    # env = make_env(env, frame_stack = False, scale = False)
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

def gen_data_atari(num_trajs, traj_len=1000, obs_num=3):
    data = {
        'trajs': []
    }
    env_params = {}
    env_params["crop_shift"] = 10
    env_params["restrict_actions"] = 4
    env_id = "PongNoFrameskip-v4"
    # env = gym.make(env_id)
    # env = make_env(env, frame_stack = False, scale = False)
    env = make_atari(env_id)
    env = wrap_deepmind(env, **env_params)
    env = wrap_pytorch(env)
    encoder = model_setup(env_id, env, False, None, True, True, 1)
    encoder.features.load_state_dict(torch.load("vanila_model.pth"))
    autoencoder = torch.load("autoencoder")
    from tqdm import tqdm
    for _ in tqdm(range(num_trajs)):
        traj_data = gen_traj_atari(encoder, autoencoder, traj_len, obs_num)
        data['trajs'].append(traj_data)

    return data
