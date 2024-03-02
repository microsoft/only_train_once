import torch.nn as nn

class DemoNetGroupNormCase1(nn.Module):
    def __init__(self):
        super(DemoNetGroupNormCase1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.gn_1 = nn.GroupNorm(32, 64)
        self.gn_2 = nn.GroupNorm(32, 128)
        self.gn_3 = nn.GroupNorm(32, 256)
        self.gn_4 = nn.GroupNorm(32, 512)
        
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leakyrelu(self.gn_1(self.conv1(x)))
        x = self.leakyrelu(self.gn_2(self.conv2(x)))
        x = self.leakyrelu(self.gn_3(self.conv3(x)))
        x = self.leakyrelu(self.gn_4(self.conv4(x)))
        return x