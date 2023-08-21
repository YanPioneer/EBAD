import random

import torch
import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, device):
        # self.max_size = 100000
        self.max_size = buffer_size
        self.p = 0
        self.size = 0  # 记录加入了多少条数据

        self.Buffer = []

        self.device = device

    def add(self, transition):
        # print(self.p)
        if self.size == self.max_size:
            self.Buffer[self.p] = transition

            self.p = (self.p + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
        else:
            self.Buffer.append(transition)

            self.p = (self.p + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=32):
        # ind = np.random.randint(0, self.size, batch_size)
        # print(ind)
        # return self.Buffer[ind].to(self.device), None, None
        return random.sample(self.Buffer, batch_size), None, None

    def clear(self):
        self.Buffer = []


