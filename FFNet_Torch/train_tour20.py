#import paramiko
import numpy as np
import matplotlib.pyplot as plt
from QAgent import Agent
import scipy.io as sio
import os

# 800 epochs for TVSum: paper
epoch = 1000

#savepath = '/data/Guha/FFNet/TorchModel/'
savepath = '/b_test/suparna/model/'
# define Q learning agent
agent = Agent(savepath)
# train_folder = '/data/Tour20/features/train/'
# gt_folder = '/data/Tour20/summary/train/'

train_folder = '/ds2/tour20/features/train/'
gt_folder = '/ds2/tour20/summary/train/'


track_epoch_acc = []
track_epoch_reward = []
track_epoch_coverage = []
best_reward = 0.0

# run total no of epoch times
for ep_num in range(epoch):
    # sum up correct predictions for all videos in every epoch
    epoch_reward = 0.0
    epoch_coverage = 0.0

    for file in os.listdir(train_folder):
        framePath = os.path.join(train_folder,file)
        temp1 = sio.loadmat(framePath)
        features = temp1['features']
        # initialize agent with current video frames and corresponding labels
        key = file.split('.mat')[0]

        # seg = '/data/Tour20/Tour20-Segmentation/'+file[0:2]+'/frm_num_'+key
        #
        # with open(seg) as segment:
        #     last = int(segment.readlines()[-1])
        #     if( last != len(features)):
        #         print (key,last," "+str(len(features)))
        # continue
        temp2 = sio.loadmat(os.path.join(gt_folder, file))
        gt = temp2['gt']
        # print (key, len(gt), " " + str(len(features)))
        # continue
        #
        # invoke agent to learn. this is called for each video in each epoch
        agent.data_init(features, gt)
        #print (key+'init done')
        agent.episode_run()

        mean_reward = agent.total_reward / agent.steps
        print("video:{} reward is {}% ".format(file, mean_reward))
        epoch_reward += mean_reward

    track_epoch_reward.append(epoch_reward)
    print( 'epoch - ' + str(ep_num) + ' epoch mean reward -' + str(epoch_reward))

    # save the model once we get better epoch rewards than previous
    if (epoch_reward > best_reward):
        best_reward = epoch_reward
        agent.save_model('torch_tour20_{}.model'.format(ep_num))
        print('saved model at epoch', ep_num)

    # plot and save epoch_history graph after each 5 epochs
    if ((ep_num % 25 == 0) or (ep_num==epoch-1)):

        fig2 = plt.figure()
        ax = fig2.add_subplot(111)
        x = np.arange(len(track_epoch_reward))

        ax.plot(x, track_epoch_reward, color="r")
        ax.set_xlabel("No of Epoch", color="black")
        ax.set_ylabel("epoch_reward", color="black")
        ax.tick_params(axis='x', colors="b")
        ax.tick_params(axis='y', colors="b")

        #fig2.savefig('tour20_out/tour20_epoch_reward_{}.png'.format(ep_num))
        fig2.savefig('/b_test/suparna/output/tour20_epoch_reward_{}.png'.format(ep_num))