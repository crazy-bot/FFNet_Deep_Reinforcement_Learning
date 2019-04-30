import numpy as np
import torch
import matplotlib.pyplot as plt
from QAgent import Agent
import scipy.io as sio
import os
import network as net



test_folder = '/ds2/tour20/features/test/'
gt_folder = '/ds2/tour20/summary/test/'

modelPath = '/b_test/suparna/model/tour20/torch_tour20_954.model'
model = net.FFNet()
model.load_state_dict(torch.load(modelPath))
model.eval()


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
    summary = np.zeros(frame_num)
    id_curr = 0

    while id_curr < frame_num:
        state = torch.FloatTensor(features[id_curr]).unsqueeze(0).view(1, 4096)
        action_index = torch.max(model(state), dim=1)[1].cpu()
        id_next = id_curr + action_index + 1
        if id_next > frame_num - 1:
            break
        summary[id_next] = 1
        id_curr = id_next

    # sio.savemat('/data/Guha/Tour20/Result/'+key,{'summary':summary})

    # calculate evaluation metric

    segpath = '/ds2/tour20/Tour20-Segmentation/' + key[0:2] + '/frm_num_' + key

    with open(segpath) as segment:
        data = segment.readlines()
        start = 0
        impshot_count = 0
        for i in range(0, len(data), 2):
            if(i+1 >= len(data)):
               break
            stop = int(data[i + 1])
            # check if the shot is important
            # if (gt[0][start] == 1):
            #     matches = np.sum(summary[start:stop] == gt[0][start:stop])
            #     impshot_count += 1
            #     for hit in range(1, len(shots_coverage) + 1):
            #         if (matches >= hit):
            #             shots_coverage[hit - 1] += 1
            # start = stop

            ############## irrespective of important shots ###############
            matches = np.sum(summary[start:stop])
            start = stop
            impshot_count += 1

            for hit in range(1, len(shots_coverage) + 1):
                if (matches > hit):
                    shots_coverage[hit - 1] += 1

        shots_coverage = shots_coverage / impshot_count

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
#plt.show()
fig1.savefig('/b_test/suparna/test.png')
#fig1.savefig('tour20_out/test.png')


