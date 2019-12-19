import torch
import torch.nn as nn


class PerceptionModule(nn.Module):
    def __init__(self):
        super(PerceptionModule, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),
            
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, x):
        out = self.features(x)
        return out

    
class MeasurementModule(nn.Module):
    def __init__(self):
        super(MeasurementModule, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(3, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, x):
        out = self.features(x)
        return out
    

class GoalModule(nn.Module):
    def __init__(self):
        super(GoalModule, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(3 * 6, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, x):
        out = self.features(x)
        return out
    

class ExpectationBlock(nn.Module):
    def __init__(self, action_size):
        super(ExpectationBlock, self).__init__()

        self.action_size = action_size

        self.features = nn.Sequential(
            nn.Linear(512 + 128 + 128, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 3 * 6),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        out = self.features(x).expand(self.action_size, -1, -1).contiguous().view(batch_size, -1).squeeze()
        return out
    

class ActionBlock(nn.Module):
    def __init__(self, action_size):
        super(ActionBlock, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(512 + 128 + 128, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 3 * 6 * action_size),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(3 * 6 * action_size)
        )
        
    def forward(self, x):
        out = self.features(x).squeeze()
        return out
    
    
class Model(nn.Module):
    def __init__(self, action_size):
        super(Model, self).__init__()

        self.perception_module = PerceptionModule()
        self.measurement_module = MeasurementModule()
        self.goal_module = GoalModule()
        
        self.expectation_block = ExpectationBlock(action_size)
        self.action_block = ActionBlock(action_size)
        
    def forward(self, s, m, g):
        perception_out = self.perception_module(s)
        measurement_out = self.measurement_module(m)
        goal_out = self.goal_module(g)
        
        concat_layer = torch.cat([perception_out, measurement_out, goal_out], dim=-1)
        expectation_block = self.expectation_block(concat_layer)
        action_block = self.action_block(concat_layer)

        out = expectation_block + action_block
        
        return out