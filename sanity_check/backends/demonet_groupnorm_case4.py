import torch.nn as nn
import torch

class DemoNetGroupNormCase4(nn.Module):
    def __init__(self):
        super(DemoNetGroupNormCase4, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(256, 72, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.gn_1 = nn.GroupNorm(32, 128)
        self.gn_2 = nn.GroupNorm(32, 256)
        self.gn_3 = nn.GroupNorm(32, 256)
        self.gn_4 = nn.GroupNorm(32, 256)
        self.gn_5 = nn.GroupNorm(32, 256)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leakyrelu(self.gn_1(self.conv1(x)))
        x_1 = torch.cat([x, x], dim=1)
        # print(x_1.shape)
        x_2 = self.gn_3(self.conv2(self.gn_2(x_1)))
        # print(x_2.shape)
        x_3 = torch.cat([self.conv3(x_1), x], dim=1)
        x_3 = self.conv4(self.gn_4(x_3))
        x_3 = self.gn_5(x_3)
        # print(x_3.shape)
        x = self.conv5(x_2 + x_3)
        return x

# model = DemoNetGroupNormCase4()
# x = torch.rand(1, 3, 32, 32)
# print(model(x).shape)