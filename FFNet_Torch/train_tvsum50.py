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

savepath = '/b_test/suparna/model/tvsum50'
train_folder = '/ds2/tvsum50/features/train/'
gt_path = '/ds2/tvsum50/data/gt_score.txt'
#train_folder = '/data/Guha/FFNet/tour20/train'

# define Q learning agent
agent = Agent(savepath)

########### Read the ground truth ##################
## gt_score.txt - key value pair . key - video name, value = binary 1/0 for all the frames
gt_dict = {}
shots_dict = {}
with open(gt_path) as fp:
    for line in fp:
        val = line.rstrip().split(' ')
        values = [float(elem) for elem in val[1:]]
        shots_dict[val[0]] = values[0]
        gt_dict[val[0]] = values[1:]
# print('ground truth', gt_dict)

track_epoch_acc = []
track_epoch_reward = []
track_epoch_coverage = []
best_reward = 0.0

# run total no of epoch times
for ep_num in range(epoch):
    # sum up correct predictions for all videos in every epoch
    epoch_reward = 0.0
    epoch_coverage = 0.0
    # At each episode select all video
    for file in os.listdir(train_folder):
        framePath = os.path.join(train_folder,file)
        mat = sio.loadmat(framePath)
        features = mat['Features']
        # initialize agent with current video frames and corresponding labels
        key = file.split('_alex')[0]
        # print('features '+key,len(features),len(gt_dict[key]))
        # continue
        agent.data_init(features, [gt_dict[key]])

        # invoke agent to learn. this is called for each video in each epoch
        agent.episode_run()

        # calculate evaluation metric
        # shots_coverage = 0
        # frames_per_shot = int(shots_dict[key])
        # no_of_shots = round(len(features) / frames_per_shot)
        # for i in range(0, len(features), frames_per_shot):
        #     if (i + frames_per_shot >= len(features) - 1):
        #         end = len(features) - 1
        #     else:
        #         end = i + frames_per_shot - 1
        #     matches = np.sum(agent.selection[i:end] == agent.gt[0][i:end])
        #     if (matches > 10):
        #         shots_coverage += 1
        # mean_coverage = shots_coverage / no_of_shots
        # print ("video:{} coverage is {}% at threshold 10".format(file, mean_coverage))
        # epoch_coverage += mean_coverage

        # print correct prediction for current video
        # correct_pred = np.sum(agent.selection == agent.labels)
        # epoch_corrects += correct_pred
        # total_frames += len(frames)
        # print 'epoch - '+str(i)+ ' video - '+subdir+' Correct Prediction-',(float(correct_pred)/len(frames))*100

        mean_reward = agent.total_reward / agent.steps
        print("video:{} reward is {}% ".format(file, mean_reward))
        epoch_reward += mean_reward

    # print correct prediction for every epoch
    # epoch_acc = (float(epoch_corrects) / total_frames) * 100
    # track_epoch_acc.append(epoch_acc)
    # print 'epoch - '+str(i)+ ' epoch mean correct frames -'+ str(epoch_acc)

    track_epoch_reward.append(epoch_reward)
    print( 'epoch - ' + str(ep_num) + ' epoch mean reward -' + str(epoch_reward))


    # track_epoch_coverage.append(epoch_coverage)
    # print('epoch - ' + str(ep_num) + ' epoch coverage -' + str(epoch_coverage))

    # save the model once we get better epoch rewards than previous
    if (epoch_reward > best_reward):
        best_reward = epoch_reward
        agent.save_model('torch_tvsum50_{}.model'.format(ep_num))
        print('saved model at epoch', ep_num)

    # plot and save epoch_history graph after each 5 epochs
    if (ep_num == 0 or ep_num % 5 == 0):
        # fig1 = plt.figure()
        # ax = fig1.add_subplot(111)
        # x = np.arange(len(track_epoch_coverage))
        #
        # ax.plot(x, track_epoch_coverage, color="r")
        # ax.set_xlabel("No of Epoch", color="black")
        # ax.set_ylabel("epoch_coverage", color="black")
        # ax.tick_params(axis='x', colors="b")
        # ax.tick_params(axis='y', colors="b")
        #
        # fig1.savefig('DQN_Out/torch_epoch_coverage_{}.png'.format(ep_num))

        fig2 = plt.figure()
        ax = fig2.add_subplot(111)
        x = np.arange(len(track_epoch_reward))

        ax.plot(x, track_epoch_reward, color="r")
        ax.set_xlabel("No of Epoch", color="black")
        ax.set_ylabel("epoch_reward", color="black")
        ax.tick_params(axis='x', colors="b")
        ax.tick_params(axis='y', colors="b")

        fig2.savefig('/b_test/suparna/output/tvsum50/reward_{}.png'.format(ep_num))

