from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F
## hidden_layer_size = 20

class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        l1 = self.relu(self.fc1(x.float()))
        l2 = self.relu(self.fc2(l1))
        l3 = self.relu(self.fc3(l2))
        l4 = self.fc4(l3)
        return l4
        
def get_network_input(player, apple):
    proximity = player.getproximity()
    x = torch.cat([torch.from_numpy(player.pos).double(), torch.from_numpy(apple.pos).double(), 
                   torch.from_numpy(player.dir).double(), torch.tensor(proximity).double()])
    return x

class QNetwork_CNN(nn.Module):        
    def __init__(self, w, h, output_dim):
        super(QNetwork_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w), stride=1))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h), stride=1))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, output_dim)
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
        
def get_network_input_for_CNN(player, apple):
    x = torch.zeros((3, 17, 17))
    for i in player.prevpos:
        x[1][int(i[0])+1][int(i[1])+1] = 1
    x[0][int(apple.pos[0])+1][int(apple.pos[1])+1] = 1
    x[2][int(player.pos[0])+1][int(player.pos[1])+1] = 1
    x[1][int(player.pos[0])+1][int(player.pos[1])+1] = 0
    return x

class QNetwork_CNN_with_Dir(nn.Module):        
    def __init__(self, w, h, output_dim):
        super(QNetwork_CNN_with_Dir, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w), stride=1))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h), stride=1))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size+4, output_dim)
        
    def forward(self, x):
        x = x.view((-1, x.shape[-1]))
        CNN_feature = x[:, :-4].view((-1, 3, 17, 17))
        dir_feature = x[:, -4:]
        CNN_feature = F.relu(self.bn1(self.conv1(CNN_feature)))
        CNN_feature = F.relu(self.bn2(self.conv2(CNN_feature)))
        CNN_feature = F.relu(self.bn3(self.conv3(CNN_feature)))
        return self.head(torch.cat((CNN_feature.view(CNN_feature.size(0), -1), dir_feature), axis=1))
        
def get_network_input_for_CNN_with_Dir(player, apple):
    x = torch.zeros((3, 17, 17))
    for i in player.prevpos:
        x[1][int(i[0])+1][int(i[1])+1] = 1
    x[0][int(apple.pos[0])+1][int(apple.pos[1])+1] = 1
    x[2][int(player.pos[0])+1][int(player.pos[1])+1] = 1
    x[1][int(player.pos[0])+1][int(player.pos[1])+1] = 0
    y = torch.zeros((4))
    y[int(player.dir[0]*2+player.dir[1])] = 1
    return torch.cat((x.view((-1)), y))