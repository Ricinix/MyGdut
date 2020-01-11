import string

import torch
from torch import nn

characters = '-' + string.digits + string.ascii_uppercase + string.ascii_lowercase


def decode(sequence):
    a = ''.join([characters[x] for x in sequence])
    print("a: ", a)
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j + 1]])
    print("s: ", s)
    if len(s) == 0:
        return ''
    if a[-1] != characters[0] and s[-1] != a[-1]:
        s += a[-1]
    return s


class MyModel(nn.Module):
    def __init__(self, kernel_layer):
        super(MyModel, self).__init__()
        self.kernel_layer = kernel_layer

    def forward(self, x):
        output = self.kernel_layer(x)
        output_argmax = output.permute(1, 0, 2).argmax(dim=-1)
        return output_argmax[0]


def save(model, version_code):
    model.eval()
    example = torch.rand(1, 3, 60, 140)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("./model3-%s.pt" % version_code)
