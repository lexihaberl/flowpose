import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights

class BasicCNN(nn.Module):
    def __init__(self, dropout_p = 0.5):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding='same')
        self.batch1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding='same')
        self.batch2 = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(32*32*32, 256)
        self.batch = nn.BatchNorm1d(256)
        # 3 for translation and 6 for rotation based on "On the Continuity of Rotation Representations in Neural Networks"
        self.fc2 = nn.Linear(256, 9)
        
    def forward(self, x):
        x1 = self.pool(F.gelu(self.batch1(self.conv1(x))))
        x2 = self.pool(F.gelu(self.batch2(self.conv2(x1))))
        x4 = self.dropout(self.flatten(x2))
        x5 = F.gelu(self.batch(self.fc1(x4)))
        out = self.fc2(x5)
        transl = out[:, :3]
        rot_r1 = out[:, 3:6]
        rot_r2 = out[:, 6:9]
        R1 = rot_r1 / torch.norm(rot_r1, dim=1).view(-1, 1)
        R3 = torch.cross(R1, rot_r2)
        R3 = R3 / torch.norm(R3, dim=1).view(-1, 1)
        R2 = torch.cross(R3, R1)
        rotmat = torch.stack([R1, R2, R3], dim=2)
        out = torch.cat([transl, rotmat.flatten(start_dim=1)], dim=1)
        return out
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
        return self

class Resnet18(nn.Module):
    def __init__(self, obj_diameter):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights = weights) 
        self.model.fc = nn.Linear(512, 9)
        self.obj_diameter = obj_diameter
    
    def forward(self, x):
        out = self.model(x)
        transl = out[:, :3] * self.obj_diameter
        rot_r1 = out[:, 3:6]
        rot_r2 = out[:, 6:9]
        R1 = rot_r1 / torch.norm(rot_r1, dim=1).view(-1, 1)
        R3 = torch.cross(R1, rot_r2)
        R3 = R3 / torch.norm(R3, dim=1).view(-1, 1)
        R2 = torch.cross(R3, R1)
        rotmat = torch.stack([R1, R2, R3], dim=2)
        out = torch.cat([transl, rotmat.flatten(start_dim=1)], dim=1)
        return out
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        return self