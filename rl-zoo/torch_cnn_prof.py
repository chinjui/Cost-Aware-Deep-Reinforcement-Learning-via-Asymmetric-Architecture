# ===============================================================================
# Training example: python torch_cnn_prof.py CarRacing-v0
# ===============================================================================
import numpy as np
import gym
import sys
import torch
from thop import profile
import torch.nn as nn
import torch.nn.functional as F

if len(sys.argv) != 2:
  raise "You need to provide `env_id` as an argument"

env_id = sys.argv[1]
env = gym.make(env_id)
ob_space = env.observation_space
ac_space = env.action_space
num_hid_layers = 2
# hid_size = int(sys.argv[2])
print('ob_space: {}, ac_space: {}'.format(ob_space, ac_space))
print('ob_space: {}, ac_space: {}'.format(ob_space.shape, ac_space.shape))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.fc1 = nn.Linear(ob_space.shape[0], hid_size)
        # self.fc2 = nn.Linear(hid_size, hid_size)
        # self.out = nn.Linear(hid_size, ac_space.shape[0]*2)
        self.cnn1 = nn.Conv2d(in_channels=ob_space.shape[2], out_channels=32, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

    def forward(self, x):
        x = F.relu(self.cnn1(x))
        # x = F.relu(self.cnn2(x))
        # x = F.relu(self.cnn3(x))
        return x



# model = models.Sequential()
# model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=ob_space.shape))
# model.add(layers.Conv2D(64, (4, 4), strides=(2, 2),  activation='relu'))
# model.add(layers.Conv2D(64, (3, 3), strides=(1, 1),  activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(512))

model = Model()
inputs = torch.randn(1, ob_space.shape[2], *ob_space.shape[0:2])
inputs.dim()
macs, params = profile(model, inputs=(inputs, ))
print("flops: {}".format(macs*2))


