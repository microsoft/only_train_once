import torch.nn as nn
import torch

class DemoNetGroupNormCase3(nn.Module):
    def __init__(self):
        super(DemoNetGroupNormCase3, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(768, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.gn_1 = nn.GroupNorm(32, 128)
        self.gn_2 = nn.GroupNorm(32, 256)
        self.gn_3 = nn.GroupNorm(32, 384)
        self.gn_4 = nn.GroupNorm(32, 768)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leakyrelu(self.gn_1(self.conv1(x)))
        x = torch.cat([x, x], dim=1)
        x = self.gn_2(x)
        x = torch.cat([self.gn_3(self.conv2(x)), self.conv3(x)], dim=1)
        x = self.gn_4(x)
        x = self.conv4(x)
        return x

