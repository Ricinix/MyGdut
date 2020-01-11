import string
from collections import OrderedDict

import get_verify_code
import save_model
import torch
from captcha.image import ImageCaptcha
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms.functional import to_tensor


class Model(nn.Module):
    def __init__(self, n_classes, input_shape=(3, 64, 128)):
        super(Model, self).__init__()
        self.input_shape = input_shape
        channels = [32, 64, 128, 256, 256]
        layers = [2, 2, 2, 2, 2]
        kernels = [3, 3, 3, 3, 3]
        pools = [2, 2, 2, 2, (2, 1)]
        modules = OrderedDict()

        def cba(name, in_channels, out_channels, kernel_size):
            modules[f'conv{name}'] = nn.Conv2d(in_channels, out_channels, kernel_size,
                                               padding=(1, 1) if kernel_size == 3 else 0)
            modules[f'bn{name}'] = nn.BatchNorm2d(out_channels)
            modules[f'relu{name}'] = nn.ReLU(inplace=True)

        last_channel = 3
        for block, (n_channel, n_layer, n_kernel, k_pool) in enumerate(zip(channels, layers, kernels, pools)):
            for layer in range(1, n_layer + 1):
                cba(f'{block + 1}{layer}', last_channel, n_channel, n_kernel)
                last_channel = n_channel
            modules[f'pool{block + 1}'] = nn.MaxPool2d(k_pool)
        modules[f'dropout'] = nn.Dropout(0.25, inplace=True)

        self.cnn = nn.Sequential(modules)
        self.lstm = nn.LSTM(input_size=self.infer_features(), hidden_size=128, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(in_features=256, out_features=n_classes)

    def infer_features(self):
        x = torch.zeros((1,) + self.input_shape)
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x.shape[1]

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


def load_model():
    return torch.jit.load('model2-script-colab.pt')


def test_data():
    gen = captcha_generate.GenerateCaptcha()
    X, y = gen.gen_test_captcha()
    print(X.shape)
    plt.imshow(X.view(60, 140, 3))
    print(gen.decode_captcha(y))
    plt.show()


def generate_model():
    characters = '-' + string.digits + string.ascii_lowercase
    width, height, n_len, n_classes = 140, 60, 4, len(characters)
    model = Model(n_classes, input_shape=(3, height, width))
    save_model.save(model, "test")


def test_model():
    img = get_verify_code.get_pic()
    get_verify_code.show_pic(img)
    input = to_tensor(img)
    model = load_model()
    y = model(input.view(1, 3, 60, 140))
    print(y.shape)
    print(y.permute(1, 0, 2).shape)
    print(y.permute(1, 0, 2).argmax(dim=-1))
    print("code:", save_model.decode(y.permute(1, 0, 2).argmax(dim=-1)[0]))


def transform_model():
    model = load_model()
    example = torch.rand(1, 3, 60, 140)
    traced_script_module = torch.jit.trace(model.kernel_layer, example)
    traced_script_module.save("./model-%s.pt" % "cuda2")


if __name__ == '__main__':
    # generate_model()
    test_model()
    # transform_model()
