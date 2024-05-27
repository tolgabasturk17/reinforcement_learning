import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        # fully connected layers
        # nn.Linear tamamen bağlantılı katmanı temsil eder. Bu katman girdi boyutlarını
        # çıktı boyutlarına lineer bir şekilde dönüştürür. Yapay sinir ağlarında
        # genellikle giriş, gizli ve çıkış katmanı olarak kullanılır.
        self.fc1 = nn.Linear(*input_dims, 128)         # input layer
        self.fc2 = nn.Linear(128,128)       # hidden layer 128 input and 128 oputput
        self.fc3 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

