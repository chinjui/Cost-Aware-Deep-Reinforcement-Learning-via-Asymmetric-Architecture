import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import gym
import argparse
import os
from matplotlib.lines import Line2D
import dmc2gym

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', type=str, default='', help='Folder which contains statistic_file.txt and rgb_arrays.pickle')
args = parser.parse_args()

folder = args.folder
assert folder != ''
folder_split = folder.split(os.sep)
if folder_split[-1] == '':
  del folder_split[-1]
env_id = folder_split[-2]
try:
  print("env:", env_id)
  env = gym.make(env_id)
except:
  domain_task = env_id.split('-')
  env = dmc2gym.make(domain_name=domain_task[0], task_name=domain_task[1], seed=1)
video_name = folder_split[-3]
video_name += '-' + folder_split[-1]

statistic_file = os.path.join(folder, 'statistic_file.txt')
rgb_file = os.path.join(folder, 'rgb_arrays.pickle')
path = os.path.normpath(folder)
action_file = os.path.join(folder, 'actions.pickle')
actions = np.load(action_file, allow_pickle=True)
rgbs = np.load(rgb_file, allow_pickle=True)
print("shape of actions:", actions.shape)

l_color = 'yellow'        # color for large policy scatter dots
s_color = 'lightyellow'   # color for small policy scatter dots

# read macros, rewards per step
with open(statistic_file, 'r') as f:
  for i, line in enumerate(f):
    if i == 0:
      score = '-score%.1f' % float(line.split(' ')[1])
      video_name += score
    if i == 2:
      macro_acts = line.split(' ')
      for i in range(len(macro_acts)):
        try:
          macro_acts[i] = int(macro_acts[i])
        except:
          del macro_acts[i]
      macro_acts = np.array(macro_acts)
    if i == 5:
      rewards = line.split(' ')
      for i in range(len(rewards)):
        try:
          rewards[i] = float(rewards[i])
        except:
          del rewards[i]
      rewards = np.array(rewards)

print("len of (macro_acts, rewards): %d, %d" % (len(macro_acts), len(rewards)))

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='', artist='',
                comment='')
writer = FFMpegWriter(fps=15, metadata=metadata)

plt.style.use('dark_background')
fig = plt.figure()
fig_n_rows = actions.shape[1] + 1
if fig_n_rows <= 6:
  fig_n_rows = 6
window_size = 60

# title: policy chosen and reward
title_ax = plt.subplot2grid((fig_n_rows, 2), (0, 0), rowspan=1)
title_ax.axis('off')
title = title_ax.text(0.5, 0.5, 'policy: %s, reward: %.3f' % (0, 0), horizontalalignment='center', verticalalignment='center')

# RGBs from env
img_ax = plt.subplot2grid((fig_n_rows, 2), (2, 0), rowspan=fig_n_rows - 2)
img_ax.axis('off')
run_img = img_ax.imshow(rgbs[0])

# reward scatter
rew_ax = plt.subplot2grid((fig_n_rows, 2), (1, 0), rowspan=1)
rew_ax.set_title('Rewards')
pad = (max(rewards) - min(rewards)) / 20
rew_ax.set_ylim([min(rewards)-pad, max(rewards)+pad])
rew_scatter = rew_ax.scatter([], [], s=6)

# legend
legend_ax = plt.subplot2grid((fig_n_rows, 2), (0, 1), rowspan=1)
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Small policy', markerfacecolor=s_color, markeredgecolor=s_color, markersize=6),
                   Line2D([0], [0], marker='o', color='w', label='Large policy', markerfacecolor=l_color, markeredgecolor=l_color, markersize=6)]
matplotlib.rcParams['legend.handlelength'] = 0
# rew_ax.legend(handles=legend_elements, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', mode='expand', ncol=2)
legend_ax.legend(handles=legend_elements, loc='center', mode='expand', ncol=2)
legend_ax.axis('off')

axes = []
act_lines = []
for i in range(actions.shape[1]):
  ax = plt.subplot2grid((fig_n_rows, 2), (i+1, 1))
  axes.append(ax)
  act_lines.append(ax.scatter([], [], s=6))
  ax.axhline(y=0, color='r', linestyle='-')
  pad = (env.action_space.high[i] - env.action_space.low[i]) / 20
  ax.set_ylim([env.action_space.low[i] - pad, env.action_space.high[i] + pad])
  if i != actions.shape[1] - 1:   # disable xaxis if not the bottom ax
    ax.get_xaxis().set_visible(False)
axes[0].set_title('Actions')
plt.tight_layout(pad=0.7)

x = np.arange(actions.shape[0])
with writer.saving(fig, "%s.mp4" % video_name, 150):
  for i in x:   # from 0 to episode_len
    left = (i + 1) - window_size
    left = left if left > 0 else 0
    x0 = x[left:i+1]
    color = np.where(macro_acts[left:i+1]==1, l_color, s_color)
    for j, ax in enumerate(axes):
      y0 = actions[left:i+1, j]
      # act_lines[j].set_data(x0, y0)
      act_lines[j].set_offsets(np.stack([x0, y0], axis=1))
      act_lines[j].set_color(color)
      ax.set_xlim([x0[-1]-window_size, x0[-1]])
    # handles, labels = act_lines[0].legend_elements(prop="colors", alpha=0.6)

    r0 = rewards[left:i+1]
    rew_scatter.set_offsets(np.stack([x0, r0], axis=1))
    rew_scatter.set_color(color)
    rew_ax.set_xlim([x0[-1]-window_size, x0[-1]])

    run_img.set_data(rgbs[i])
    img_ax.set_aspect('equal')

    policy = 'large' if macro_acts[i] == 1 else 'small'
    title.set_text('policy: %s, reward: %.3f' % (policy, rewards[i]))

    writer.grab_frame()
