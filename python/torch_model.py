import torchvision
from torch import nn

import captcha_generate


class ConvNet(nn.Module):

    def __init__(self, output_size):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=2),  # in:(bs,3,60,160)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2),  # out:(bs,32,30,80)

            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2),  # out:(bs,64,15,40)

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2)  # out:(bs,64,7,20)
        )

        self.fc1 = nn.Linear(7616, 500)
        self.fc2 = nn.Linear(500, output_size)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # reshape to (batch_size, 64 * 7 * 30)
        output = self.fc1(x)
        output = self.fc2(output)

        return output


def get_model() -> nn.Module:
    model = torchvision.models.resnet18()
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    gen = captcha_generate.GenerateCaptcha()
    # model = ConvNet(gen.get_parameter()[4] * gen.get_parameter()[2])
    model.fc = nn.Linear(model.fc.in_features, gen.get_parameter()[4] * gen.get_parameter()[2])
    return model


if __name__ == '__main__':
    import torch

    input = torch.rand(1, 1, 60, 140)
    net = get_model()
    net(input)
