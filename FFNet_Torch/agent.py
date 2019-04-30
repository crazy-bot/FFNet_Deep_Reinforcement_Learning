import numpy as np
import random
import network as net
import torch
from collections import namedtuple

Transition = namedtuple('Transition',
                            ('state', 'action','reward', 'next_state'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.transition = []
        self.position = 0

    def add(self, *args):
    # transition: state, action,reward, target
        if len(self.transition) < self.capacity:
            self.transition.append(None)
        self.transition[self.position] = Transition(*args)

        self.position = (self.position+1) % self.capacity

        # self.transition.append(Transition(*args))


    def sample(self, batch_size):
        rand_idx = random.randint(0, len(self.transition)-batch_size)
        return self.transition[rand_idx:rand_idx+batch_size-1]
        # return random.sample(self.transition,batch_size)

    def get_size(self):
    	return len(self.transition)

    def reset(self):
    	del self.transition[:]

class Agent(object):
    def __init__(self, alpha, gamma, actionSpace):
        # decay: future reward decay rate
        self.decay_rate = gamma

        # learning rate
        #self.learning_rate = alpha

        # explore: exploration rate
        self.explore_low = 0.1
        self.explore_decay = .00001
        self.explore_rate = 1

        self.actionSpace = actionSpace

        # agents estimate actions value for the current state
        self.Q_eval = net.FFNet(alpha)
        # agents estimate the actions value for the next state
        self.Q_next = net.FFNet(alpha)

        #starting training from 465 epoch and last model saved was 423 yet.
        self.Q_eval.load_state_dict(torch.load('/data/Guha/FFNet/New_Model/tvsum50_423.model'))
        self.Q_next.load_state_dict(self.Q_eval.state_dict())

        # memory size is set at 128 transitions : paper
        self.memory = ReplayMemory(capacity=128)

        self.steps = 0

        # how manty times the agent calls the learn functions - used for target network replacement
        self.learn_step_counter = 0
        # how often we want to replace the target network
        self.replace_target_cnt = 100

        self.selection = []

        self.total_reward = 0.0

    def initVariables(self,frames,labels):
        self.frames = frames
        self.labels = np.asarray(labels)
        self.selection = []
        self.total_reward = 0.0
        self.steps = 0


    # select an action based on policy
    def chooseAction(self, state):

        # Return random floats in the half-open interval [0.0, 1.0)
        rand = np.random.random()
        # explore time
        if (rand > 1 - self.explore_rate):
            action_index = np.random.choice(self.actionSpace, 1)[0]
        else:
            #state = torch.Tensor(state).unsqueeze(0).view(1,3,224,224)
            #print('exploit')
            state = torch.FloatTensor(state).unsqueeze(0).view(1,4096)
            action_index = torch.max(self.Q_eval.forward(state),dim=1)[1].cpu()

        return action_index

    # compute the reward
	# REWARD = -skip penalty + hit reward
    # skip penalty: penalty for skipping the interval
    # hit reward: reward for hitting important frame

    def reward(self, id_curr, id_next):
        ############ skip penalty calculation ###############
        skip_seg = self.labels[id_curr + 1:id_next]
        # skip_minus: penalty for skipping important frames
        skip_minus = sum(skip_seg)
        # skip_plus: reward for skipping unimportant frames
        skip_plus = len(skip_seg) - skip_minus
        # skip_penalty: combination of above two with trade-off factor 0.8. normalized by largest skip action(25)
        skip_penalty = (skip_minus - 0.8 * skip_plus) / 25

        ############## Hit Reward Calculation ##################
        # hit reward: sum over frames within window left and right distributed as gaussian
        hit_reward = 0.0
        #gaussian_value = [0.0001,0.0044,0.0540,0.2420,0.3989,0.2420,0.0540,0.0044,0.0001]
        window_size = 4
        gaussian_dist = np.linspace(0.1,1,num=window_size+1) # [0.1  , 0.325, 0.55 , 0.775, 1.   ]
        no_of_frames = len(self.labels)

        # reward calculated over left window
        for i in range(1,window_size+1):
            if(id_next-i > -1) and self.labels[id_next-i]==1:
                hit_reward +=  gaussian_dist[window_size-i]

        # reward calculated over right window
        for i in range(1,window_size+1):
            if(id_next+i < no_of_frames) and self.labels[id_next+i]==1:
                hit_reward +=  gaussian_dist[window_size-i]

        # reward for the current frame
        if self.labels[id_next] == 1:
            hit_reward += gaussian_dist[window_size ]

        reward = (skip_penalty + hit_reward)
        return reward

    def run_episode(self,batch_size):
        no_of_frames = len(self.frames)
        # selection array contains 1 for important frames as predicted
        self.selection = np.zeros(len(self.labels))
        id_curr = 0
        self.selection[id_curr ] = 1

        # loop through all the frames
        while (id_curr < no_of_frames):
            # choose an action given the state
            action = self.chooseAction(self.frames[id_curr])
            # apply the action and get the next state
            id_next = id_curr + action.item() + 1

            if id_next > (no_of_frames-1):
                print('predicted next frame {} is out of range {}'.format(id_next,no_of_frames-1))
                break
            # set id next as important frame
            try:
                self.selection[id_next] = 1
            except Exception as e:
                print(e)
                print('predicted next frame {} is out of range {}'.format(id_next, no_of_frames - 1))
                break

            # steps count on choosing one action
            self.steps += 1

            # calculate reward on takling this action
            reward = self.reward(id_curr,id_next)
            self.total_reward += reward
            nextState = self.frames[id_next]

            # add the transition
            self.memory.add(self.frames[id_curr],action,reward,nextState)

            # batch learning to break correlations between state transition
            if self.memory.get_size() >= batch_size:
                #print('training')
                self.optimize_model(batch_size)

                #added below to reset memory after each training step as per the paper.
                #self.memory.reset()

                # check if its time to update the target network by eval network
                if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
                    #  load at state to the target network froma dict converting eval network to dict
                    self.Q_next.load_state_dict(self.Q_eval.state_dict())

            # decrease the epsilon over time.
            if self.explore_rate - self.explore_decay > self.explore_low:
                self.explore_rate -= self.explore_decay
            else:
                self.explore_rate = self.explore_low

            id_curr = id_next


    def optimize_model(self,batch_size):
        # zero out gradients for each batch as in pytorch gradients get accumulated step to step
        # if we dont do it will turn in to full learning
        self.Q_eval.optim.zero_grad()

        # we got batch size. now random subsample of this size from our memory
        sample = self.memory.sample(batch_size)
        # sample = self.memory.transition

        batch = Transition(*zip(*sample))
        # sample = np.asarray(sample)

        state_batch = torch.FloatTensor(np.stack(batch.state)).to(self.Q_eval.device) # (32*3*227*227)
        action_batch = torch.LongTensor(batch.action).to(self.Q_eval.device) # (32)
        reward_batch = torch.FloatTensor(batch.reward).to(self.Q_eval.device) # (32)
        nextState_batch = torch.FloatTensor(np.stack(batch.next_state)).to(self.Q_eval.device) # (32*3*227*227)


        # we get the action predictions for current state.Q_eval takes only state and gives output value for every state-action pair (32*25)
        # # We choose the Q  value based on action taken (action_batch). QPred_current: gathers value along the action column indexed by action_batch
        QPred = self.Q_eval(state_batch) #[ 32*25 ]
        #Index tensor must have same dimensions as input tensor.unsqueeze adds a dimension to match
        QPred_current = QPred.gather(1,action_batch.unsqueeze(1)).to(self.Q_eval.device) # [32*1]


        # Q value for next state.# # Detach variable from the current graph since we don't want gradients for next Q to propagated
        QNext = self.Q_next(nextState_batch).detach()
        #Compute max next Q value based on which action gives max Q values
        QNext_max = torch.max(QNext, dim=1, keepdim=True)[0].to(self.Q_eval.device)

        # Compute the target of the current Q values
        QTarget = reward_batch.unsqueeze(1) + self.decay_rate * QNext_max

        # loss is measured from error between current and newly expected Q values
        loss = self.Q_eval.loss(QPred_current,QTarget).to((self.Q_eval.device))
        # back propagate the loss
        loss.backward()
        # optimizer step() method updates the parameters, once the gradients are computed
        self.Q_eval.optim.step()

        #increment learn step counter. it has been used to decide when to replace the target betwork by eval network.
        self.learn_step_counter += 1
