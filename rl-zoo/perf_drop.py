import sys
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default="")
parser.add_argument("--macro-len", type=int, default=5)
parser.add_argument("--macro-hidden-size", type=int, default=32)
# parser.add_argument("--sub-hidden-sizes", nargs="*", type=int, default=[8, 64])
args = parser.parse_args()


assert args.file != ""

sns.set(style="darkgrid")
exp_name = args.file.split('/')[-3]
env_id = args.file.split('/')[-2]
print("Expirience ID: %s, Env ID: %s" % (exp_name, env_id))
sub_hidden_sizes = {
    "MountainCarContinuous-v0": [8, 64],
    "BipedalWalker-v3": [64, 256],
    "Swimmer-v3": [8, 256],
    "Ant-v3": [64, 256],
    "FetchPickAndPlace-v1": [32, 128],
    "walker-stand": [8, 64],
    "finger-spin": [8, 64],

    "HalfCheetah-v3": [8, 64],
    "Walker2d-v3": [64, 256],
    "FetchPush-v1": [8, 64],
    "FetchSlide-v1": [64, 256],
    "cartpole-swingup": [8, 64],
    "ball_in_cup-catch": [8, 64],
    "hopper-stand": [64, 256],
    "reacher-easy": [8, 64],
    "finger-turn_easy": [8, 256]

    }

args.sub_hidden_sizes = sub_hidden_sizes[env_id]

small_scores = {
    "MountainCarContinuous-v0": -11.6,
    "BipedalWalker-v3": 69.2,
    "Swimmer-v3": 35.5,
    "Ant-v3": 1690,
    "FetchPickAndPlace-v1": 0.35,
    "walker-stand": 330,
    "finger-spin": 32.9,
    }
random_scores = {
    "HalfCheetah-v3": -292.10,
    "Swimmer-v3": -0.189,
    "Walker2d-v3": 1.625,
    "Ant-v3": -50.882,
    "Hopper-v3": 14.11,
    "InvertedDoublePendulum-v2": 59.09,
    "Reacher-v2": -42.32,
    "FetchPush-v1": 0.068,
    "FetchPickAndPlace-v1": 0.034,
    "FetchSlide-v1": 0,

    "cartpole-swingup": 25.8734,
    "ball_in_cup-catch": 62.95,
    "finger-spin": 1.04,
    "fish-swim": 74.27,
    "walker-stand": 129.10,
    "BipedalWalker-v3": -101.4,
    "MountainCarContinuous-v0": -33.19,
    "hopper-stand": 2.7,
    "reacher-easy": 65.0,

                }
scores = {
          # returns
          "HalfCheetah-v3":     {8: 1560, 64: 7490,   256: 8105},
          "Swimmer-v3":         {8: 32.3, 64: 46,     256: 91.7},
          "Walker2d-v3":        {8: 581,  64: 1660,   256: 4690},
          "Ant-v3":             {8: -23,  64: 1720,   256: 4350},
          "Hopper-v3":          {8: 390,  64: 3310,   256: 3060},
          "Humanoid-v3":        {8: 77,   64: 3050,   256: 5110},
          "HumanoidStandup-v2": {8: 1.2e5,64: 1.32e5, 256: 1.97e5},
          "Reacher-v2":         {8: -12.8,64: -5.3,   256: -4.7},
          "InvertedDoublePendulum-v2": {8: 6000, 64: 9190, 256: 9140},

          "cartpole-swingup": {8: 158, 64: 805},
          "ball_in_cup-catch": {8: 129, 64: 971},
          "finger-spin": {8: 26.5, 64: 924},
          "fish-swim": {8: 78.4, 64: 118, 256: 183},
          "walker-stand": {8: 237, 64: 971},
          "BipedalWalker-v3": {64: 69.2, 256: 290.7},
          "MountainCarContinuous-v0": {8: -24, 64: 92},
          "hopper-stand": {64: 223.8, 256: 901.6},
          "reacher-easy": {8: 166.5, 64: 948.5},


          # success rates
          "FetchPush-v1":         {8: 0.07, 64: 0.98, 256: 1.00},
          "FetchPickAndPlace-v1": {8: 0.05, 64: 0.48, 128: 0.98, 256: 1.00},
          "FetchSlide-v1":        {8: 0.03, 64: 0.15, 256: 0.76},
          }

# policy costs (in flops) of policies with 2 hidden layers
# costs = {8:  595,
#          32: 3907,
#          64: 11907,
#          256: 145923}
#
costs = {
    "MountainCarContinuous-v0": {8: 195, 32: 2307, 64: 8707},
    "BipedalWalker-v3": {32: 4099, 64: 12291, 256: 147459},
    "Swimmer-v3": {8: 323, 32: 2819, 256: 137219},
    "Ant-v3": {32: 10179, 64: 24451, 256: 196099},
    "FetchPickAndPlace-v1": {32: 4163, 128: 43267},
    "walker-stand": {8: 707, 32: 4355, 64: 12803},
    "finger-spin": {8: 339, 32: 2883, 64: 9859},

    "HalfCheetah-v3": {8: 595, 32: 3907, 64: 11907},
    "Walker2d-v3": {32: 3907, 64: 11907, 256: 145923},
    "FetchPush-v1": {8: 755, 32: 4547, 64: 13187},
    "FetchSlide-v1": {32: 4547, 64: 13187, 256: 151043},
    "cartpole-swingup": {8: 243, 32: 2499, 64: 9091},
    "ball_in_cup-catch": {8: 323, 32: 2819, 64: 9731},
    "hopper-stand": {32: 3523, 64: 11139, 256: 142851},
    "reacher-easy": {8: 291, 32: 2691, 64: 9475},
    }
# hidden8_256_file = "Ant-v1_hid8,256_ent1e-2_seed1179.txt"
#
# combined_returns = []
# macro_ratios = []
# policy0_returns = []
# policy1_returns = []
# need_return = False
#
# with open(hidden8_256_file) as f:
#   for line in f:
#     match = re.search('macro_acts: ([0-9\.]+)', line)
#     if match is not None:
#       macro_ratios.append(float(match.group(1)))
#       need_return = True
#
#     match = re.search('Episode .* return: ([0-9\.]+)', line)
#     if match is not None:
#       if need_return:
#         combined_returns.append(float(match.group(1)))
#         need_return = False
#
#     match = re.search('sub 0: ([0-9\.]+), sub 1: ([0-9\.]+),', line)
#     if match is not None:
#       policy0_returns.append(float(match.group(1)))
#       policy1_returns.append(float(match.group(2)))

# macro_ratios = macro_ratios[:len(combined_returns)]
# print("combined_returns;", len(combined_returns))
# print("macro_ratios:", len(macro_ratios))
# print("policy0_returns:", policy0_returns)
# print("policy1_reutnrs:", policy1_returns)
# policy0_returns = 3501.64
large_policy_score = scores[env_id][args.sub_hidden_sizes[1]]
random_score = random_scores[env_id]
# random_score = small_scores[env_id]
# use the last 300 results
# combined_returns: returns of combined macro policy (i.e. macro, small, large)
with open(args.file, 'r') as f:
  data = []
  for i, line in enumerate(f):
    if i in [1, 4, 7]:
      # 1: macro_ratios, 4: returns, 7: success rates
      data.append(line.split(' '))
      for j in range(len(data[-1])):
        try:
          data[-1][j] = float(data[-1][j])
        except Exception as e:
          print(e, "data[%d][%d]: [%s]" % (len(data)-1, j, data[-1][j]))
          del data[-1][j]

if len(data) == 3:
  print("Length (macro, return, success): %d, %d, %d" % (len(data[0]), len(data[1]), len(data[2])))
elif len(data) == 2:
  print("Length (macro, return): %d, %d" % (len(data[0]), len(data[1])))
else:
  raise "Error"
macro_ratios = data[0]
combined_returns = data[1] if len(data) == 2 or (len(data) == 3 and len(data[2]) == 0) else data[2] # does this file contain success rates?
macro_ratios = macro_ratios[-500:]
combined_returns = combined_returns[-500:]


macro_cost = costs[env_id][args.macro_hidden_size]
policy_costs = [costs[env_id][args.sub_hidden_sizes[0]], costs[env_id][args.sub_hidden_sizes[1]]]
costs = [(ratio * policy_costs[1] + (1-ratio) * policy_costs[0] + (1 / args.macro_len) * macro_cost) / policy_costs[1] for ratio in macro_ratios]

# fig, axes = plt.subplots(ncols=2)

# plot 1: performance v.s. costs
if env_id == '':
  relative_perf = combined_returns
else:
  relative_perf = [r / large_policy_score for r in combined_returns]
  relative_perf = [(r-random_score) / (large_policy_score-random_score) for r in combined_returns]
d = pd.DataFrame(data={'Perf (0 ~ large policy score)': relative_perf, 'Costs (%)': costs})
g = sns.jointplot('Costs (%)', 'Perf (0 ~ large policy score)', data=d, color="m", kind='scatter', ratio=3, marginal_kws=dict(bins=10))
# g = sns.jointplot('Costs (%)', 'Perf (0 ~ large policy score)', data=d, color="m", kind='scatter', ratio=3)
g.ax_joint.set_ylabel('')
g.ax_joint.set_xlabel('')
g.ax_joint.set_xlim([0, 1.1])
g.ax_joint.set_ylim([-0.1, 1.5])
# g.ax_joint.set_ylim([-0.3, 1.5])
# axes[0].set_xlim([0, 1])
# axes[0].set_ylim([0, 1])
# plt.ylim(min(0, min(relative_perf)), max(1, max(relative_perf)))
# plt.xlim(0, 1)

# plot 2: both %
# relative_perf = [(r-policy0_returns) / (policy1_returns-policy0_returns) for r in combined_returns]
# d = pd.DataFrame(data={'Perf (small policy ~ large policy score)': relative_perf, 'Costs (%)': costs})
# g = sns.regplot('Perf (small policy ~ large policy score)', 'Costs (%)', data=d, color="m", ax=axes[1])
# plt.xlim(0, 1)
# plt.ylim(0, 1)
textstr = exp_name + '\n'
# textstr = textstr + "small policy: %.2f, large policy: %.2f\n" % (policy0_returns, policy1_returns)
# textstr += "costs: %.2f : %.2f" % (policy_costs[0], policy_costs[1])
# plt.suptitle(textstr, y=0.03)

# plt.show()
plt.savefig(fname="perf_cost_%s.pdf" % env_id)
