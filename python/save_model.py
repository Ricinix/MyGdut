import torch
from torch import nn
from torchvision.transforms.functional import to_tensor
import demo_cuda

class MyModel(nn.Module):
    def __init__(self, kernel_layer):
        self.kernel_layer = kernel_layer

    def forward(self, x):
        a = to_tensor(x)
        output = self.kernel_layer(a)
        output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
        return demo_cuda.decode(output_argmax[0])


def save(model, version_code):
    model.eval()
    my_model = MyModel(model)
    example = torch.rand(1, 3, 60, 140)
    traced_script_module = torch.jit.trace(my_model, example)
    traced_script_module.save("./model-%f.pt" % version_code)
