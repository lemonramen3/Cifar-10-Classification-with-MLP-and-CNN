# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class BatchNorm1d(nn.Module):
    # TODO START
    def __init__(self, num_features):
        super(BatchNorm1d, self).__init__()
        self.num_features = num_features
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Parameters
        self.weight = Parameter(torch.randn(num_features, dtype=torch.float32), requires_grad=True)
        self.bias = Parameter(torch.randn(num_features, dtype=torch.float32), requires_grad=True)

        # Store the average mean and variance
        self.register_buffer('running_mean', torch.zeros(num_features).to(device))
        self.register_buffer('running_var', torch.ones(num_features).to(device))

        # Initialize your parameter
        self.eps = 1e-5
        self.momentum = 0.1

    def forward(self, input):
        # input: [batch_size, num_feature_map * height * width]
        if self.training:
            mean = input.mean(dim=0)
            var = input.var(dim=0, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            bn_init = (input - mean) / torch.sqrt(var + self.eps)
        else:
            bn_init = (input - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return self.weight * bn_init + self.bias

# TODO END


class Dropout(nn.Module):
    # TODO START
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p
        self.keep = 1 - p

    def forward(self, input):
        # input: [batch_size, num_feature_map * height * width]
        f = (torch.rand(input.shape, device=input.device) <= self.keep).float()
        input = f * input / self.keep
        return input


# TODO END

class Model(nn.Module):
    def __init__(self, drop_rate=0.5):
        super(Model, self).__init__()
        # TODO START
        # Define your layers here
        self.fc1 = nn.Linear(3 * 32 * 32, 3 * 16 * 16)
        self.bn1 = BatchNorm1d(3 * 16 * 16)
        self.relu = nn.ReLU()
        self.dropout = Dropout(p=drop_rate)
        self.fc2 = nn.Linear(3 * 16 * 16, 10)
        # TODO END
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # TODO START
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.training:
            x = self.dropout(x)
        x = self.fc2(x)
        # the 10-class prediction output is named as "logits"
        logits = x

        # TODO END

        pred = torch.argmax(logits, 1)  # Calculate the prediction result
        if y is None:
            return pred
        y = y.long()
        loss = self.loss(logits, y)
        correct_pred = (pred.int() == y.int())
        acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

        return loss, acc
