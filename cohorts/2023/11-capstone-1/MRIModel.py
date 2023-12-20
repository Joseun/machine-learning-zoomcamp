#!/usr/bin/python3
""" MODULE FOR MRI MODEL ARHICTECTURE """
import torch
import torch.nn as nn
import torch.nn.functional as F


class MRIModel(nn.Module):
    def __init__(self):
        super().__init__()
        C_in = 1
        init_f = 16
        self.conv1 = nn.Conv2d(C_in, init_f, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(
            init_f + C_in, 2 * init_f, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(3 * init_f + C_in, 4 * init_f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(7 * init_f + C_in, 8 * init_f, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            15 * init_f + C_in, 16 * init_f, kernel_size=3, padding=1
        )
        self.dropout = nn.Dropout(p=0.15)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 4)
        self.fc6 = nn.Linear(16, 3)

    def forward(self, x):
        identity = F.avg_pool2d(x, 4, 4)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)

        identity = F.avg_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)

        identity = F.avg_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)

        identity = F.avg_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)

        x = F.relu(self.conv5(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.flatten(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.dropout(F.relu(self.fc4(x)))
        y = self.fc5(x)
        z = F.log_softmax(self.fc6(x), dim=1)
        return torch.concatenate((y, z), axis=1)
