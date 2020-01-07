import torch
from torch import nn
from torchvision.transforms.functional import to_tensor

class MyModel(nn.Module):
    def __init__(self, kernel_layer):
        self.kernel_layer = kernel_layer

    def forward(self, x):
        a = to_tensor(x)
        return self.kernel_layer(a)


def save(model, version_code):
    model.eval()
    my_model = MyModel(model)
    example = torch.rand(1, 3, 60, 140)
    traced_script_module = torch.jit.trace(my_model, example)
    traced_script_module.save("./model-%f.pt" % version_code)
