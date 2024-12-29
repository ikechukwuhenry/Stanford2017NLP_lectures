import torch
import torch.nn as nn
import torch.optim as optim
from NaiveCustomLSTM_module import NaiveCustomLSTM

vocab_length = 100

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_length + 1, 32)
        self.lstm = NaiveCustomLSTM(32, 32)
        self.fc1 = nn.Linear(32, 2)

    def forward(self, x):
        x_ = self.embedding(x)
        x_, (h_n, c_n) = self.lstm(x_)
        x_ = (x_[:, -1, :])
        x_ = self.fc1(x_)
        return x_
    

# data preprocessing
# code to be implemented later

# create and load model to GPU
device = torch.device('mps')
classifier = Net().to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.005)
loss_fn = nn.CrossEntropyLoss() 
