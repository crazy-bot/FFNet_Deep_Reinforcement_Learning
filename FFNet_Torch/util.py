import h5py
import csv
import  os
import cv2
import scipy.io as io
import torchvision.transforms as transforms
#import network as net
import torch
import  numpy as np
import pandas as pd
from PIL import Image
from alexnet import AlexNet


def parseScore2Bool():
    ##################### reading large mat file with h5py ########################
    input_file = '/data/Guha/FFNet/tvsum50/matlab/ydata-tvsum50.mat'
    output_file = '/data/Guha/FFNet/tvsum50/data/gt_score.txt'
    info_file = '/data/Guha/FFNet/tvsum50/data/ydata-tvsum50-info.tsv'
    with h5py.File(input_file) as mat, open(info_file) as info, open(output_file, 'w') as op:
        id = []
        tsvin = csv.reader(info, delimiter='\t')
        line_count = 0
        for row in tsvin:
            if line_count == 0:
                line_count = 1
                continue
            id.append(row[1])
        #print (id[0])
        tvsum50 = mat['tvsum50']
        ## keys: category gt_score length nframes title user_anno video
        nframes = tvsum50['nframes']
        video_len = tvsum50['length']
        gt_score = tvsum50['gt_score']
        gt_ref = gt_score[:, 0]  # len(gt_ref = 50
        len_ref = video_len[:, 0]
        frm_ref = nframes[:,0]

        scoreList = []
        for gt, key,len,frm in zip(gt_ref, id,len_ref,frm_ref):
            no_of_frames = int(tvsum50[frm][0])
            print (key+' '+str(no_of_frames))

            # calculate no of shots in each video. each shot is of 2 sec
            # length of video in sec
            duration = tvsum50[len][0][0]
            frames_per_shot = int(round(no_of_frames*2/duration))
            no_of_shots = int(round(no_of_frames / frames_per_shot))
            # score list
            scores = tvsum50[gt][0]
            shot_score = np.zeros(no_of_shots+1)
            #average the frame-level score to compute shot-level score
            for count,i in enumerate(range(0, no_of_frames, frames_per_shot)):
                if (i + frames_per_shot >= no_of_frames - 1):
                    end = no_of_frames - 1
                else:
                    end = i + frames_per_shot - 1
                shot_score[count] = np.average(scores[i:end])

            shot_score_idx = shot_score.argsort()[0:20]
            # make the score list binary
            frame_scores = np.zeros(no_of_frames)

            for idx in shot_score_idx:
                frame_scores[idx*frames_per_shot:(idx*frames_per_shot)+frames_per_shot] = 1

            # scoreList.append(scores)
            line = key + ' ' + str(frames_per_shot)+' '+' '.join(map(str, frame_scores))
            #print (len(scores))
            op.write(line)
            op.write('\n')
            line = ''

def extractFeatures(train_folder,dest_folder):

    transform1 = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # data_transform = transforms.Compose([
    #     transforms.Resize((224,224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])
    model = net.AlexFC7().cuda()
    model.eval()

    for subdir, dirs, files in os.walk(train_folder):
        # feat is list of all features which are expraxted from the model
        feat = []

        # need dict to save as mat file
        dict = {}
        # skip the first one. its extra taking the root path
        if subdir != '/data/Tour20/frames/MP7.mp4' :
            #noOfVid = len(dirs)
            continue
        for idx in range(0, len(files)-1):
            framePath = os.path.join(subdir, 'frame' + str(idx) + '.jpg')
            frame = cv2.imread(framePath)
            #pil = Image.open(framePath)
            if frame is None:
                continue
            frame = cv2.resize(frame, (227, 227))
            frame = transform1(frame)
            frame = frame.unsqueeze(0).view(1, 3, 227, 227)
            # pil = data_transform(pil)
            # pil = pil.unsqueeze(0).view(1, 3, 224, 224)
            out = model(frame).cpu()
            out = out.numpy().reshape(4096)
            feat.append(out)

        dict['features'] = feat

        vidKey = subdir[subdir.rfind('/') + 1:subdir.rfind('.')]
        storage_path = dest_folder + '/' + vidKey
        # if not os.path.exists(storage_path):
        #     os.makedirs(storage_path)

        io.savemat(storage_path, dict)
        print(storage_path)
        if subdir == '/data/Tour20/frames/MP7.mp4' :
            #noOfVid = len(dirs)
            break

def createTour20Label():

    ID = [ \
        'AW1', 'AW2', 'AW3', 'AW4', 'AW5', 'AW6', 'AT6' \
        'MP1', 'MP2', 'MP3', 'MP4', 'MP5', 'MP6', 'MP7' \
        'TM1', 'TM2', 'TM3', 'TM4', 'TM5', 'TM6', 'TM7' \
        'BF1', 'BF2', 'BF3', 'BF4', 'BF5', \
        'SB1', 'SB2', 'SB3', 'SB4','SB5' \
        'MC1', 'MC2', 'MC3', 'MC4', 'MC5', 'MC6', 'MC7', 'MC8','MC9','MC10' \
        'AT1', 'AT2', 'AT3', 'AT4', 'AT5','AT6' \
        'GB1', 'GB2', 'GB3', 'GB4', 'GB5','GB6' \
        'ET1', 'ET2', 'ET3', 'ET4', 'ET5', 'ET6','ET7','ET8' \
        'NC1', 'NC2', 'NC3', 'NC4', 'NC5', 'NC6','NC7','NC8' \
        'TA1', 'TA2', 'TA3', 'TA4', 'TA5','TA6' \
        'HM1', 'HM2', 'HM3', 'HM4', 'HM5','HM6' \
        'CB1', 'CB2', 'CB3', 'CB4', 'CB5','CB6' \
        'GM1', 'GM2', 'GM3', 'GM4','GM5' \
        'BK1', 'BK2', 'BK3', 'BK4', 'BK5', 'BK6', 'BK7','BK8','BK9' \
        'WP1', 'WP2', 'WP3', 'WP4','WP5' \
        'CI1', 'CI2', 'CI3', 'CI4', 'CI5', 'CI6','CI7','CI8' \
        'SH1', 'SH2', 'SH3', 'SH4', 'SH5', 'SH6', 'SH7', 'SH8','SH9','SH10' \
        'PT1', 'PT2', 'PT3', 'PT4', 'PT5', 'PT6', 'PT7','PT8','PT9' \
        'PC1', 'PC2', 'PC3', 'PC4', 'PC5','PC6']

    categories = ['AW','MP','TM','BF','SB','MC','AT','GB','ET','NC','TA','HM','CB','GM','BK','WP','CI','SH','PT','PC']

    gt = {}
    summary_folder = '/data/Tour20/Tour20-UserSummaries/'
    for i in range(1, 4):
        path = summary_folder + 'User{}'.format(i)
        for filename in os.listdir(path):
            excelpath = os.path.join(path,filename)
            # print excelpath
            data = pd.read_excel(excelpath,sheet_name = 'Sheet1')
            for key in list(data)[:-3]:
                #print key
                if(gt.has_key(key)):
                    l = gt[key]
                    l.extend(data[:][key][pd.notnull(data[:][key])].tolist())
                else:
                    l = data[:][key][pd.notnull(data[:][key])].tolist()
                gt[key] = l

    dict = {}
    for key in gt.keys():
        if key == 'TA6':
            print key
        segpath = '/data/Tour20/Tour20-Segmentation/' + key[0:2] + '/frm_num_' + key
        #segpath = '/data/Tour20/Tour20-Segmentation/AT/test'
        with open(segpath) as segment:
            data = segment.readlines()
            score = np.zeros(int(data[-1]))
            start = 0
            shot_count = 1
            for i in range(0,len(data),2):
                #start = int(data[i])
                stop=int(data[i+1])

                if (shot_count in gt[key]):
                    score[start:stop] = 1

                start = stop
                shot_count += 1

            # for shot_no,line in enumerate(data):
            #     # if shot is important
            #     if(shot_no+1 in gt[key]):
            #         start = int(line) - 1
            #         stop = int(data[shot_no+1])
            #         score[start:stop] = 1

        dict['gt'] = score
        io.savemat('/data/Tour20/summary/'+key, dict)
        print '/data/Tour20/summary/'+key


if __name__ == '__main__':
    #parseScore2Bool()

    # train_folder = '/data/Tour20/frames/'
    # dest_folder = '/data/Tour20/features/'
    # # train_folder = '/data/Guha/FFNet/tvsum50/frames/test/'
    # # dest_folder = '/data/Guha/FFNet/tvsum50/features/test/'
    # #
    # extractFeatures(train_folder,dest_folder)
    #
    # #createTour20Label()
    # print('debug')
