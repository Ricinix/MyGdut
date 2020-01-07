from time import time

import torch
import torch.nn.functional as F
from torch import nn

import captcha_generate
import save_model
import torch_model


class nCrossEntropyLoss(torch.nn.Module):

    def __init__(self, length, n=4):
        super(nCrossEntropyLoss, self).__init__()
        self.n = n
        self.length = length
        self.total_loss = 0
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, label):
        self.total_loss = 0
        output_t = output[:, 0:self.length]
        label_t = label[:, 0:self.length].argmax(dim=1)

        for i in range(self.n):
            output_t = torch.cat((output_t, output[:, self.length * i:self.length * i + self.length]), 0)
            # 损失的思路是将一张图平均剪切为4张小图即4个多分类，然后再用多分类交叉熵方损失
            label_t = torch.cat((label_t, label[:, i * self.length:i * self.length + self.length].argmax(dim=1)), 0)
            self.total_loss = self.loss(output_t, label_t)

            # self.total_loss += self.loss(output[:, i * self.length:i * self.length + self.length],
            # label[:, i * self.length:i * self.length + self.length].argmax(dim=1))
        return self.total_loss


def equal(np1, np2):
    n = 0
    for i in range(np1.shape[0]):
        if (np1[i, :] == np2[i, :]).all():
            n += 1
    print(np1.topk(k=5, dim=1)[1])
    print(np2.topk(k=5, dim=1)[1])
    print("n: %d" % n)
    return n


if __name__ == '__main__':
    captcha = captcha_generate.GenerateCaptcha()
    width, height, char_num, characters, classes = captcha.get_parameter()
    batch_size = 64

    net = torch_model.get_model()

    # 由于识别验证码本质上是对验证码中的信息进行分类，所以我们这里使用cross_entropy的方法来衡量损失。
    # 优化方式选择的是AdamOptimizer，学习率设置比较小，为1e-4，防止学习的太快而训练不好。
    trainer = torch.optim.Adam(net.parameters(), lr=0.05, weight_decay=0.1)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("初始化完成， 开始训练！")
    loss = nCrossEntropyLoss(classes, char_num)
    net.train()
    epoch = 1
    train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time()
    version = 0

    x_cor = torch.zeros(batch_size, char_num)
    for i in range(batch_size):
        x_cor[i] = i
    x_cor = x_cor.type(torch.long)
    for X, y in captcha.gen_captcha(batch_size):
        pred = torch.zeros(batch_size, 1).type(torch.long)
        if device is not None:
            X = X.to(device)
            y = y.to(device)
        trainer.zero_grad()
        y_hat = net(X)
        l = loss(y_hat, y.type(torch.long))
        l.backward()
        trainer.step()
        train_l_sum += l.item()
        index = y_hat

        for i in range(4):
            pre = F.log_softmax(y_hat[:, classes * i:classes * i + classes], dim=1)  # (bs, 10)
            pred = torch.cat((pred, pre.max(1, keepdim=True)[1] + i * classes), dim=1)
        pred = pred[:, 1:]
        mask = torch.zeros_like(y_hat)
        mask[x_cor, pred] = 1

        train_acc_sum += equal(mask.type(torch.long), y.type(torch.long))
        n += y.size(0)
        acc_percent = train_acc_sum / n
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec' % (
            epoch, train_l_sum / n, acc_percent, time() - start))
        epoch += 1
        if 0.1 <= acc_percent < 0.2 and version == 0:
            save_model.save(net, 0.1)
            version += 1
        elif 0.2 <= acc_percent < 0.3 and version == 1:
            save_model.save(net, 0.2)
            version += 1
        elif 0.3 <= acc_percent < 0.4 and version == 2:
            save_model.save(net, 0.3)
            version += 1
        elif 0.4 <= acc_percent < 0.5 and version == 3:
            save_model.save(net, 0.4)
            version += 1
        elif 0.5 <= acc_percent < 0.6 and version == 4:
            save_model.save(net, 0.5)
            version += 1
        elif 0.6 <= acc_percent < 0.7 and version == 5:
            save_model.save(net, 0.6)
            version += 1
        elif 0.7 <= acc_percent < 0.8 and version == 6:
            save_model.save(net, 0.7)
            version += 1
        elif 0.8 <= acc_percent < 0.9 and version == 7:
            save_model.save(net, 0.8)
            version += 1
        elif 0.9 <= acc_percent:
            save_model.save(net, 0.9)
            break
        if train_l_sum < 0.001:
            save_model.save(net, 0.9)
            break
