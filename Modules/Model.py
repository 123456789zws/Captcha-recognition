import torch
from torch import nn


class VerModel(nn.Module):
    def __init__(self, in_channels=3, num_class=10):
        super().__init__()
        self.in_channels = in_channels
        self.num_class = num_class
        self.ln_size = [32, 16, 16]
        # [3, 32, 32] => [6, 28, 28]
        self.block_per = nn.Sequential(
            nn.BatchNorm2d(num_features=3),
            nn.SiLU(),
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5))
        )
        # 使用一个卷积进行下降
        # [6, 28, 28] => [6, 14, 14]
        # self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        # [6, 28, 28] => [9, 24, 24]
        self.convPool_1 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=(5, 5))
        # [9, 24, 24] => [16, 20, 20]
        self.block_mid = nn.Sequential(
            nn.BatchNorm2d(num_features=9),     # 参数表示通道数
            nn.SiLU(),
            nn.Conv2d(in_channels=9, out_channels=16, kernel_size=(5, 5))
        )
        # [16, 20, 20] => [32, 16, 16]
        self.convPool_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5))
        self.linear_1 = nn.Linear(self.ln_size[0] * self.ln_size[1] * self.ln_size[2], 1024)
        self.linear_2 = nn.Linear(1024, self.num_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_tensor):
        out_tensor = self.block_per(in_tensor)
        out_tensor = self.convPool_1(out_tensor)
        out_tensor = self.block_mid(out_tensor)
        out_tensor = self.convPool_2(out_tensor)
        out_tensor = out_tensor.view(-1, self.ln_size[0] * self.ln_size[1] * self.ln_size[2])
        out_tensor = self.linear_2(self.linear_1(out_tensor))
        return self.softmax(out_tensor)


if __name__ == '__main__':
    in_tensor = torch.randn(3, 3, 32, 32)
    model = VerModel(in_channels=3, num_class=10)
    out_tensor = model(in_tensor)
    print(out_tensor)