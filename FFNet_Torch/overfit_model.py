import numpy as np
import cv2
import os
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from QAgent import Agent
import scipy.io as sio
import os


# 800 epochs for TVSum: paper
epoch = 800

savepath = '/data/Guha/FFNet/TorchModel/'

# define Q learning agent
agent = Agent(savepath)

track_epoch_acc = []
track_epoch_reward = []
track_epoch_coverage = []
best_reward = 0.0

# run total no of epoch times
for ep_num in range(epoch):
    # sum up correct predictions for all videos in every epoch
    epoch_reward = 0.0
    epoch_coverage = 0.0

    mat = sio.loadmat('/data/Tour20/features/train/AT2.mat')
    features = mat['features']
    gt = sio.loadmat('/data/Guha/FFNet/tour20/MP7_gt.mat')
    agent.data_init(features, gt['gt'])

    # invoke agent to learn. this is called for each video in each epoch
    agent.episode_run()

    mean_reward = agent.total_reward / agent.steps
    print("video:{} reward is {}% ".format(file, mean_reward))
    epoch_reward += mean_reward

    track_epoch_reward.append(epoch_reward)
    print( 'epoch - ' + str(ep_num) + ' epoch mean reward -' + str(epoch_reward))

    # save the model once we get better epoch rewards than previous
    if (ep_num == epoch-1):
        agent.save_model('overfit_tour20_{}.model'.format(ep_num))
        print('saved model at epoch', ep_num)

    # plot and save epoch_history graph after each 5 epochs
    if (ep_num != 0 and ep_num % 20 == 0):

        fig2 = plt.figure()
        ax = fig2.add_subplot(111)
        x = np.arange(len(track_epoch_reward))

        ax.plot(x, track_epoch_reward, color="r")
        ax.set_xlabel("No of Epoch", color="black")
        ax.set_ylabel("epoch_reward", color="black")
        ax.tick_params(axis='x', colors="b")
        ax.tick_params(axis='y', colors="b")

        fig2.savefig('overfit/torch_epoch_reward_{}.png'.format(ep_num))

