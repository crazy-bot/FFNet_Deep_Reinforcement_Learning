import tensorflow as tf
import numpy as np
from nn_layer import Layer
from nn import NeuralNet
from read_data import Episode
from agent import Agent
import scipy.io as sio
import os
import matplotlib.pyplot as plt

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

test_folder = '/data/Guha/tvsum50/features/test/'

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

shots_coverage = np.zeros(20)
mean_coverage = np.zeros(20)
total_shots = 0

for file in os.listdir(test_folder):
    shots_coverage = np.zeros(20)
    framePath = os.path.join(test_folder, file)
    mat = sio.loadmat(framePath)
    features = mat['Features']

    # test: no of frames and no of labels shouldbe same
    key = file.split('_alex')[0]

    # print('features '+key,len(features),len(gt_dict[key]))
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

	# name = 'output/sum_'+test_name[i%test_num]
	# sio.savemat(name,{'summary': summary})

    # calculate evaluation metric

    frames_per_shot = int(shots_dict[key])
    no_of_shots = round(len(features) / frames_per_shot)
    # total_shots += no_of_shots
    for i in range(0, len(features), frames_per_shot):
        if (i + frames_per_shot >= len(features) - 1):
            end = len(features) - 1
        else:
            end = i + frames_per_shot - 1
        matches = np.sum(summary[i:end])

        for hit in range(1,len(shots_coverage)+1):
            if (matches > hit):
                shots_coverage[hit-1] += 1
    shots_coverage = shots_coverage/no_of_shots

    mean_coverage += shots_coverage

# mean_coverage = (shots_coverage / total_shots)
mean_coverage = mean_coverage/9
print ("mean coverage is {}".format(mean_coverage))

# # plot hit number vs coverage graph
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
#fig1.savefig('output/tensor_test_tvsum50_coverage.png')

print('Test done.')
