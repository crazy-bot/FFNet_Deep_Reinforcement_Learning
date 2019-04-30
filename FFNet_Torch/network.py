import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import models



class FFNet(nn.Module):
    def __init__(self,alpha=.0002):
        super(FFNet, self).__init__()
        # # get the pretrained AlexNet model
        # alex = models.alexnet(pretrained=True)
        #
        # # Freeze the weights
        # for param in alex.parameters():
        #     param.requires_grad = False
        #
        # # remove the FC8+Relu layer
        # alex.classifier = alex.classifier[:-2]
        # # feature = list(alex.children())
        #
        # # FFnet : AlexNet Fc7 + 4 layers as in paper
        # # self.feature = nn.Sequential(*feature)
        # self.Alex_feature = nn.Sequential(*list(alex.children())[0])
        # self.Alex_classifier = nn.Sequential(*list(alex.children())[1])

        self.l1 = nn.Linear(in_features=4096, out_features=400)
        self.l2 = nn.Linear(in_features=400, out_features=200)
        self.l3 = nn.Linear(in_features=200, out_features=100)
        self.l4 = nn.Linear(in_features=100, out_features=25)

        self.relu = nn.ReLU()


        self.optim = optim.RMSprop(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()

        self.ave_value = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, input):
        input = input.to(self.device)

        # Net to predict action
        X = input.view(input.size(0), -1)
        X = self.relu(self.l1(X))
        X = self.relu(self.l2(X))
        X = self.relu(self.l3(X))
        actions = self.l4(X)

        return actions

class AlexFC7(nn.Module):
    def __init__(self):
        super(AlexFC7, self).__init__()
        # get the pretrained AlexNet model
        alex = models.alexnet(pretrained=True)

        # # Freeze the weights
        # for param in alex.parameters():
        #     param.requires_grad = False
        #
        # # remove the FC8+Relu layer
        # alex.classifier = alex.classifier[:-2]
        # # feature = list(alex.children())
        #
        # # FFnet : AlexNet Fc7 as in paper
        # # self.feature = nn.Sequential(*feature)
        # self.Alex_feature = nn.Sequential(*list(alex.children())[0])
        # self.Alex_classifier = nn.Sequential(*list(alex.children())[1])

        new_classifier = nn.Sequential(*list(alex.classifier.children())[:-2])
        alex.classifier = new_classifier
        for param in alex.parameters():
            param.requires_grad = False

        self.model = alex

    def forward(self, input):
        input = input.cuda()

        # Net to extract features of frame
        # ft = self.Alex_feature(input)
        # ft = ft.view(ft.size(0), -1)
        # ft = self.Alex_classifier(ft)
        ft = self.model(input)
        return ft
AlexFC7()