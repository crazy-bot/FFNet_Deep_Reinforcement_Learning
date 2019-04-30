import numpy as np
import cv2
import os
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from QAgent import Agent
import scipy.io as sio
import os
import network as net


test_folder = '/data/Guha/tvsum50/our features/train'
# test_folder = '/data/Guha/FFNet/tour20/train/'
modelPath = 'model/torch_tvsum50_763.model'
model = net.FFNet()
model.load_state_dict(torch.load(modelPath))
model.eval()
# gt_score.txt - key value pair . key - video name, value = binary 1/0 for all the frames
gt_dict = {}
shots_dict = {}
with open('/data/Guha/tvsum50/data/gt_score.txt') as fp:
    for line in fp:
        val = line.rstrip().split(' ')
        values = [float(elem) for elem in val[1:]]
        shots_dict[val[0]] = values[0]
        gt_dict[val[0]] = values[1:]
#print('ground truth', gt_dict)

shot_count = 0
mean_coverage = np.zeros(20)
total_shots = 0
for file in os.listdir(test_folder):
    shots_coverage = np.zeros(20)
    framePath = os.path.join(test_folder, file)
    mat = sio.loadmat(framePath)
    # gt = sio.loadmat('/data/Guha/FFNet/tour20/MP7_gt.mat')
    features = mat['features']
    # gt = gt['gt']
    # test: no of frames and no of labels shouldbe same
    key = file.split('.mat')[0]

    # print('features '+key,len(features),len(gt_dict[key]))
    # continue
    frame_num = len(features)
    prediction = np.zeros(frame_num)
    id_curr = 0
    #prediction[id_curr] = 1

    while id_curr < frame_num:
        state = torch.FloatTensor(features[id_curr]).unsqueeze(0).view(1, 4096)
        action_index = torch.max(model(state), dim=1)[1].cpu()
        id_next = id_curr + action_index + 1
        if id_next > frame_num - 1:
            break
        prediction[id_next] = 1
        id_curr = id_next

    # calculate evaluation metric

    frames_per_shot = int(shots_dict[key])
    # no_of_shots = round(len(features) / frames_per_shot)
    # total_shots += no_of_shots
    shot_count = 0
    for i in range(0, len(features), frames_per_shot):

        if (i + frames_per_shot >= len(features) - 1):
            end = len(features) - 1
        else:
            end = i + frames_per_shot - 1

        ############### check if the shot is important ################
        if(gt_dict[key][i] == 1):
            matches = np.sum(prediction[i:end] == gt_dict[key][i:end])
            #matches = np.sum(prediction[i:end])
            shot_count += 1

            for hit in range(1,len(shots_coverage)+1):
                if (matches >= hit):
                    shots_coverage[hit-1] += 1

        # matches = np.sum(prediction[i:end])
        # shot_count += 1
        # for hit in range(1, len(shots_coverage) + 1):
        #     if (matches >= hit):
        #         shots_coverage[hit - 1] += 1


    shots_coverage = shots_coverage/shot_count

    mean_coverage += shots_coverage

# mean_coverage = (shots_coverage / total_shots)
mean_coverage = mean_coverage/max(mean_coverage)
print ("mean coverage is {}".format(mean_coverage))

# plot hit number vs coverage graph
fig1 = plt.figure()
ax = fig1.add_subplot(1,1,1)
x = np.arange(1,len(mean_coverage)+1)

ax.plot(x, mean_coverage, color="r")
ax.scatter(x, mean_coverage, color="r")
ax.set_xlabel("Hit Number", color="black")
ax.set_ylabel("mean_coverage", color="black")
ax.tick_params(axis='x', colors="b")
ax.tick_params(axis='y', colors="b")
ax.set_xticks(x)
ax.set_xlim([1,20])
ax.set_ylim([0,1])
plt.show()
fig1.savefig('tvsum50_out/test90_imp.png')


