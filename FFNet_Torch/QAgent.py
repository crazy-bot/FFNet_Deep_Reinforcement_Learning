# class of learning agent
import tensorflow as tf
import numpy as np
import network as net
import torch
import random
import math


class Memo(object):
    def __init__(self):
        self.state = []
        self.target = []

    def add(self, state, target):
        # state: previous state, the input of the Q net
        # target: the target output of the Q net given the state
        self.state.append(state)
        self.target.append(target.detach().numpy())

    def get_size(self):
        return np.shape(self.state)[0]

    def reset(self):
        # self.state = []
        # self.target = []
        del self.state[:]
        del self.target[:]


class Agent(object):
    def __init__(self, path):
        # layers: architecture of Q network
        # batch: number of observations in mini-batch set
        # explore: exploration rate
        # decay: future reward decay rate
        # path: model path

        self.batch_size = 128
        self.decay_rate = 0.8
        self.learning_rate = .0002
        self.explore_low = 0.1
        self.explore_decay = 0.00001
        self.explore_rate = 1
        self.directory = path
        self.num_action = 25 #self.Q.l4.out_features
        ##### build Q network
        self.Q = net.FFNet(self.learning_rate)

        ##### data-related variables
        self.feat = []
        self.gt = []
        self.memory = Memo()
        self.selection = []

    # initialize with data
    # def data_init(self, current_eps):
    # 	self.feat = current_eps.feat
    # 	self.gt = current_eps.gt

    def data_init(self, feat, gt):
        self.feat = feat
        self.gt = gt
        self.total_reward = 0.0
        self.steps = 0

    # select an action based on policy
    def policy(self, id_curr):
        exploration = np.random.choice(range(2), 1, p=[1 - self.explore_rate, self.explore_rate])
        # exploration==1: explore
        # exploration==0: exploit
        if exploration == 1:  # exploration
            action_index = np.random.choice(range(self.num_action), 1)[0]
            # print('              explore:  '+str(action_index))
            state = torch.FloatTensor(self.feat[id_curr]).unsqueeze(0).view(1, 4096)
            action_value = self.Q.forward(state)

            # record average Q value
           # self.Q.ave_value.append(torch.mean(action_value[0]))

            # print(action_value[0])
            return action_index
        else:  # exploitation
            state = torch.FloatTensor(self.feat[id_curr]).unsqueeze(0).view(1, 4096)
            action_value = self.Q.forward(state)

            # record average Q value
           # self.Q.ave_value.append(torch.mean(action_value[0]))
            # self.Q.ave_value = np.append(self.Q.ave_value,np.mean(action_value[0]))

            action_index = torch.argmax(action_value[0]).cpu()

            #print('exploit:  '+str(action_index))
            # print(action_value[0])
            return action_index

    # perform action to get next state
    def action(self, id_curr, a_index):
        id_next = id_curr + a_index + 1  # action 0,1,2,...
        return id_next

    # compute the reward
    # REWARD 3:reward with distribution and terms on length of skipping
    def reward(self, id_curr, a_index, id_next):

        if (len(self.gt[0]) <= id_next):
            print('list index out of range', len(self.gt[0]), id_next)

        gaussian_value = [0.0001, 0.0044, 0.0540, 0.2420, 0.3989, 0.2420, 0.0540, 0.0044, 0.0001]
        # skipping interval,missing part
        seg_gt = self.gt[0][id_curr + 1:id_next]
        total = len(seg_gt)
        n1 = sum(seg_gt)
        n0 = total - n1
        miss = (0.8 * n0 - n1) / 25  # largest action step.
        # accuracy
        acc = 0
        if id_next - 4 > -1:
            if self.gt[0][id_next - 4] == 1:
                acc = acc + 0.0001
        if id_next - 3 > -1:
            if self.gt[0][id_next - 3] == 1:
                acc = acc + 0.0044
        if id_next - 2 > -1:
            if self.gt[0][id_next - 2] == 1:
                acc = acc + 0.0540
        if id_next - 1 > -1:
            if self.gt[0][id_next - 1] == 1:
                acc = acc + 0.2420
        if self.gt[0][id_next] == 1:
            acc = acc + 0.3989
        if id_next + 1 < len(self.gt[0]):
            if self.gt[0][id_next + 1] == 1:
                acc = acc + 0.2420
        if id_next + 2 < len(self.gt[0]):
            if self.gt[0][id_next + 2] == 1:
                acc = acc + 0.0540

        if id_next + 3 < len(self.gt[0]):
            if self.gt[0][id_next + 3] == 1:
                acc = acc + 0.0044
        if id_next + 4 < len(self.gt[0]):
            if self.gt[0][id_next + 4] == 1:
                acc = acc + 0.0001
        r = miss + acc
        return r

    # update target Q value
    def update(self, r, id_curr, id_next, a_index):
        state = torch.FloatTensor(self.feat[id_curr]).unsqueeze(0).view(1, 4096)
        target = self.Q(state)
        next_state = torch.FloatTensor(self.feat[id_next]).unsqueeze(0).view(1, 4096)
        if(isinstance(r,np.float64)):
            r = r.item()
        target[0][a_index] = r + self.decay_rate * max(self.Q(next_state)[0])
        return target.cpu()

    # run an episode to get case set for training
    def episode_run(self):
        frame_num = np.shape(self.feat)[0]
        self.selection = np.zeros(frame_num)
        id_curr = 0
        self.selection[id_curr] = 1
        while id_curr < frame_num:
            a_index = self.policy(id_curr)
            id_next = self.action(id_curr, a_index)
            if id_next > frame_num - 1:
                break
            self.selection[id_next] = 1
            r = self.reward(id_curr, a_index, id_next)
            self.total_reward += r
            self.steps += 1
            target_vector = self.update(r, id_curr, id_next, a_index)[0]
            input_vector = self.feat[id_curr]
            self.memorize(input_vector, target_vector)
            if self.memory.get_size() == self.batch_size:
                # print('training')
                self.train()
            id_curr = id_next

    # training Q net using one batch data
    def train(self):
        self.explore_rate = max(self.explore_rate - self.explore_decay, self.explore_low)
        state =  torch.FloatTensor(self.memory.state).to(self.Q.device)
        target =  torch.FloatTensor(self.memory.target).to(self.Q.device)

        # zero out gradients for each batch as in pytorch gradients get accumulated step to step
        # if we dont do it will turn in to full learning
        self.Q.optim.zero_grad()

        QPred_current = self.Q(state)

        # loss is measured from error between current and newly expected Q values
        loss = self.Q.loss(QPred_current, target).to((self.Q.device))
        # back propagate the loss
        loss.backward()
        # optimizer step() method updates the parameters, once the gradients are computed
        self.Q.optim.step()
        del state
        del target
        torch.cuda.empty_cache()
        self.memory.reset()

    # store current observation to memory
    def memorize(self, state, target):
        # observation: new observation (s,a,r,s')
        self.memory.add(state, target)

    # reset data-related variables
    def data_reset(self):
        self.feat = []
        self.gt = []
        self.selection = []

    # save trained Q net
    def save_model(self, filename):
        # module backup
        path = self.directory
        torch.save(self.Q.state_dict(), path+filename)


