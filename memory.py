import torch

from random import sample


class ReplayMemory:
    def __init__(self, capacity=20000):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.s1 = torch.zeros(capacity, 4, 84, 84).to(device)
        self.s2 = torch.zeros(capacity, 4, 84, 84).to(device)
        self.a = torch.zeros(capacity).to(device)
        self.r = torch.zeros(capacity).to(device)
        self.m = torch.zeros(capacity, 3).to(device)
        self.isterminal = torch.zeros(capacity).to(device)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, m, isterminal, reward):
        idx = self.pos
        self.s1[idx,:,:,:] = s1
        self.a[idx] = action
        self.m[idx,:] = m
        if not isterminal:
            self.s2[idx,:,:,:] = s2
        self.isterminal[idx] = isterminal
        self.r[idx] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, size):
        idx = sample(range(0, self.size), size)
        return (self.s1[idx], self.a[idx], self.s2[idx], self.m[idx], self.isterminal[idx], self.r[idx])