import tensorflow as tf
import numpy as np
from nn_layer import Layer
from nn import NeuralNet
from read_data import Episode
from agent import Agent
import scipy.io as sio
import os
import matplotlib.pyplot as plt

# cv2.imread('tour20_out/tour20_epoch_reward_5.png')
# exit()

test_folder = '/data/Guha/Tour20/features/test/'
gt_folder = '/data/Guha/Tour20/summary/test/'
# define neural network layout
l1 = Layer(4096,400,'relu')
l2 = Layer(400,200,'relu')
l3 = Layer(200,100,'relu')
l4 = Layer(100,25,'linear')
layers = [l1,l2,l3,l4]
learning_rate = 0.0002
loss_type = 'mean_square'
opt_type = 'RMSprop'

Q = NeuralNet(layers,learning_rate,loss_type, opt_type)
Q.recover('model/','Q_net_all_11_0_1000')


mean_coverage = np.zeros(20)
total_shots = 0
for file in os.listdir(test_folder):
    shots_coverage = np.zeros(20)
    framePath = os.path.join(test_folder, file)
    temp1 = sio.loadmat(framePath)
    features = temp1['features']
    key = file.split('.mat')[0]
    temp2 = sio.loadmat(os.path.join(gt_folder, file))
    gt = temp2['gt']
    # test: no of frames and no of labels shouldbe same

    # print (key, len(gt), " " + str(len(features)))
    # continue
    frame_num = len(features)
    id_curr = 0
    summary = np.zeros(frame_num)
    Q_value = []


    while id_curr < frame_num:
        action_value = Q.forward([features[id_curr]])
        a_index = np.argmax(action_value[0])
        id_next = id_curr + a_index + 1
        if id_next > frame_num - 1:
            break
        summary[id_next] = 1
        Q_value.append(max(action_value[0]))
        id_curr = id_next

    # calculate evaluation metric

    segpath = '/data/Guha/Tour20/Tour20-Segmentation/' + key[0:2] + '/frm_num_' + key
    #segpath = '/data/Tour20/Tour20-Segmentation/AT/test'
    with open(segpath) as segment:
        data = segment.readlines()
        nGtSeg = len(data) / 2

        start = 1
        impshot_count = 0
        for i in range(0, len(data), 2):
            stop = int(data[i + 1])
            # check if the shot is important
            # if (gt[0][start] == 1):
            #     matches = np.sum(summary[start:stop] == gt[0][start:stop])
            #     impshot_count += 1
            #     for hit in range(1, len(shots_coverage) + 1):
            #         if (matches > hit):
            #             shots_coverage[hit - 1] += 1

            #matches = np.sum(summary[start:stop] == gt[0][start:stop])
            matches = np.sum(summary[start:stop]) #as per the reply from author
            start = stop
            impshot_count += 1

            for hit in range(1, len(shots_coverage) + 1):
                 if (matches >= hit):
                    shots_coverage[hit - 1] += 1

        shots_coverage = shots_coverage / nGtSeg


        mean_coverage += shots_coverage
    #print ("{} shot coverage is {}".format(key, shots_coverage/count_imp_shots))



# mean_coverage = (mean_coverage / total_shots)
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
#fig1.savefig('test_tour20_full_coverage.png')


