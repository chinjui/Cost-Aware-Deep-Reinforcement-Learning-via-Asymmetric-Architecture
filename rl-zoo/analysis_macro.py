import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import scipy.misc
from skimage.transform import resize
from skimage.util import img_as_ubyte
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', type=str, default='', help='Folder which contains statistic_file.txt and rgb_arrays.pickle')
args = parser.parse_args()

# folder = 'savedir/swimmer_hid8,256_ent1e-2_analysis/episode101/'
folder = args.folder
assert folder != ''
statistic_file = os.path.join(folder, 'statistic_file.txt')
rgb_file = os.path.join(folder, 'rgb_arrays.pickle')
action_file = os.path.join(folder, 'actions.pickle')
path = os.path.normpath(folder)
video_name = path.split(os.sep)[2]
video_name += '_' + path.split(os.sep)[4]

rgbs = np.load(rgb_file, allow_pickle=True)
actions = np.load(action_file, allow_pickle=True)

with open(statistic_file, 'r') as f:
  for i, line in enumerate(f):
    if i == 0:
      score = '_score%.1f' % float(line.split(' ')[1])
      video_name += score
    if i == 2:
      macro_acts = line.split(' ')
      macro_acts_repeat = []
      for v in macro_acts:
        macro_acts_repeat.append(v)
    if i == 5:
      rewards = line.split(' ')
      try:
        for i in range(len(rewards)):
          rewards[i] = float(rewards[i])
      except:
        pass

print(len(macro_acts_repeat), len(rewards))
# font type
font = cv2.FONT_HERSHEY_SIMPLEX
org = (25, 25)
fontScale = 0.5
color = (255, 0, 0)
thickness = 1

out = cv2.VideoWriter('%s.mp4' % video_name, cv2.VideoWriter_fourcc(*'MP4V'), 15, rgbs[0].shape[0:2])
for i, img in enumerate(rgbs):
  # Using cv2.putText() method
  img = cv2.putText(img, 'macro: %s, reward: %.3f' % (macro_acts_repeat[i], rewards[i]), org, font,
                     fontScale, color, thickness, cv2.LINE_AA)
  out.write(img)
out.release()
