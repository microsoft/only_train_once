import torch.nn as nn
import torch

class DemoNetGroupNormCase2(nn.Module):
    def __init__(self):
        super(DemoNetGroupNormCase2, self).__init__()
        # self.conv1 = nn.Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.conv2 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        # self.gn_1 = nn.GroupNorm(32, 128)
        # self.gn_2 = nn.GroupNorm(32, 256)

        self.conv1 = nn.Conv2d(3, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.gn_1 = nn.GroupNorm(4, 12)
        self.gn_2 = nn.GroupNorm(4, 24)
        
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leakyrelu(self.gn_1(self.conv1(x)))
        x = torch.cat([x, x], dim=1)
        x = self.gn_2(x)
        x = self.conv2(x)
        return x
    
model = DemoNetGroupNormCase2()

x = torch.rand(1, 3, 32, 32)
model(x)

