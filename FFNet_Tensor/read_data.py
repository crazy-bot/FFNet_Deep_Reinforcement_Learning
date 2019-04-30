# select a video for an episode. can be modified to randomly select.
import scipy.io as sio
import numpy as np

class Episode(object):

    def __init__(self, idx, num, name, feat_path,gt_path):
    # idx: episode index
    # num: total number of videos - 113
    # name: video name
    # path: video and gt path
        self.index = idx
        self.name = name[idx%num]
        self.feat = []
        self.gt = []
        temp_1= sio.loadmat(feat_path+self.name+'_alex_fc7_feat.mat')
        self.feat = temp_1['Features']
        temp_2 = sio.loadmat(gt_path+self.name+'_gt.mat')
        self.gt = temp_2['gt']

    #################### feature matching check ###############################
    ############# input: author's feat, input2: our feat

        feat1 = sio.loadmat('input/MP7_alex_fc7_feat.mat')['Features']

        feat2 = sio.loadmat('input2/MP7_alex_fc7_feat.mat')['features']

        match = feat1 == feat2
        print (match)

        # out1= sio.loadmat('output/sum_MP7.mat')['summary']
        # out2 = sio.loadmat('output/sum2_MP7.mat')['summary']



    def get_size(self):
        num_frames = np.shape(self.feat)[0]
        return num_frames
