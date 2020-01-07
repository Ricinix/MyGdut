import random
import string
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captcha.image import ImageCaptcha
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

characters = '-' + string.digits + string.ascii_uppercase + string.ascii_lowercase
width, height, n_len, n_classes = 140, 60, 4, len(characters)
n_input_length = 12
print(characters, width, height, n_len, n_classes)


class CaptchaDataset(Dataset):
    def __init__(self, characters, length, width, height, input_length, label_length):
        super(CaptchaDataset, self).__init__()
        self.characters = characters
        self.length = length
        self.width = width
        self.height = height
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # 随机选取4个字符
        random_str = ''.join([random.choice(self.characters[1:]) for j in range(self.label_length)])
        # 生成对应的图片并转换成tensor
        image = to_tensor(self.generator.generate_image(random_str))
        # 长度为4，每个位置为4个字符在[characters]中对应的index
        target = torch.tensor([self.characters.find(x) for x in random_str], dtype=torch.long)
        # 将scalar：input_length变成一个tensor
        input_length = torch.full(size=(1,), fill_value=self.input_length, dtype=torch.long)
        # 将scalar:：n_len变成一个tensor
        target_length = torch.full(size=(1,), fill_value=self.label_length, dtype=torch.long)
        return image, target, input_length, target_length


# 数据集
dataset = CaptchaDataset(characters, 1, width, height, n_input_length, n_len)
image, target, input_length, label_length = dataset[0]
# 输出第一个数据看看效果
print(''.join([characters[x] for x in target]), input_length, label_length)
# to_pil_image(image)

# 批量大小
batch_size = 64
# 训练集
train_set = CaptchaDataset(characters, 1000 * batch_size, width, height, n_input_length, n_len)
# 验证集
valid_set = CaptchaDataset(characters, 100 * batch_size, width, height, n_input_length, n_len)
# 将dataset转化成dataloader来读取数据
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=12)
valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=12)


# 卷积神经网络+循环神经网络
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


# 传入验证码图片的大小，获取模型
model = Model(n_classes, input_shape=(3, height, width))
inputs = torch.zeros((32, 3, height, width))
outputs = model(inputs)
# 看看32批量的输出是怎么样的
print("output shape: ", outputs.shape)
model = Model(n_classes, input_shape=(3, height, width))
model = model.cuda()
# 输出模型
print(model)


# 将序列转换为验证码答案
def decode(sequence):
    a = ''.join([characters[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j + 1]])
    if len(s) == 0:
        return ''
    if a[-1] != characters[0] and s[-1] != a[-1]:
        s += a[-1]
    return s


# 转换为验证码答案
def decode_target(sequence):
    return ''.join([characters[x] for x in sequence]).replace(' ', '')


# 计算正确率
def calc_acc(target, output):
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    target = target.cpu().numpy()
    output_argmax = output_argmax.cpu().numpy()
    # 将索引转换为验证码答案来判断对错
    a = np.array([decode_target(true) == decode(pred) for true, pred in zip(target, output_argmax)])
    return a.mean()


# 训练
def train(model, optimizer, epoch, dataloader):
    model.train()
    loss_mean = 0
    acc_mean = 0
    with tqdm(dataloader) as pbar:
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
            # 数据dataloader
            data, target = data.cuda(), target.cuda()

            # 梯度清零
            optimizer.zero_grad()
            # 向前传播
            output = model(data)

            # 计算softmax（将每一个值映射到0到1）
            output_log_softmax = F.log_softmax(output, dim=-1)
            # 计算loss
            loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)

            # 反向传播
            loss.backward()
            # 调整权重矩阵
            optimizer.step()

            # 保存loss值
            loss = loss.item()
            # 计算正确率
            acc = calc_acc(target, output)

            if batch_index == 0:
                loss_mean = loss
                acc_mean = acc

            # 指数加权移动平均
            loss_mean = 0.1 * loss + 0.9 * loss_mean
            acc_mean = 0.1 * acc + 0.9 * acc_mean

            # 更新进度条
            pbar.set_description(f'Epoch: {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')


# 验证（虽说验证不需要优化器）
def valid(model, optimizer, epoch, dataloader):
    model.eval()
    with tqdm(dataloader) as pbar, torch.no_grad():
        loss_sum = 0
        acc_sum = 0
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
            data, target = data.cuda(), target.cuda()

            output = model(data)
            output_log_softmax = F.log_softmax(output, dim=-1)
            loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)

            loss = loss.item()
            acc = calc_acc(target, output)

            loss_sum += loss
            acc_sum += acc

            loss_mean = loss_sum / (batch_index + 1)
            acc_mean = acc_sum / (batch_index + 1)

            pbar.set_description(f'Test : {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')


# Adam优化器（8说了，遇事不决上Adam）
optimizer = torch.optim.Adam(model.parameters(), 1e-3, amsgrad=True)
# 30次迭代
epochs = 30
for epoch in range(1, epochs + 1):
    train(model, optimizer, epoch, train_loader)
    valid(model, optimizer, epoch, valid_loader)
# 降低优化器的学习率，进行后15次的优化
optimizer = torch.optim.Adam(model.parameters(), 1e-4, amsgrad=True)
epochs = 15
for epoch in range(1, epochs + 1):
    train(model, optimizer, epoch, train_loader)
    valid(model, optimizer, epoch, valid_loader)

# 训练完成，保存模型
model.eval()
import save_model

save_model.save(model, "cuda")
# 一直输出答案，直到输出不一样为止
do = True
output = model(image)
output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
while do or decode_target(target) == decode(output_argmax[0]):
    do = False
    image, target, input_length, label_length = dataset[0]
    print('true:', decode_target(target))

    output = model(image.unsqueeze(0).cuda())
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    print('pred:', decode(output_argmax[0]))
# to_pil_image(image)
