"""
torch==2.4.0+cu124    torchvision==0.19.0+cu124
Python 3.12.2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# 调用官方的数据集
train_ds = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_ds  = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl  = torch.utils.data.DataLoader(test_ds, batch_size=32)

num_classes=10

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取网络  channel为1是因为MNIST数据集不是彩色图片
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)        # 步长默认等于核的大小
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)

        # 分类网络
        self.fc1 = nn.Linear(1600,64)
        self.fc2 = nn.Linear(64,num_classes)

    #前向传播
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)   #从第 1 维开始展平，保留第 0 维（通常是批量大小）
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

"""
    1, 28, 28 -> 32, 26, 26           26 = (28+2*p-k)/s+1 = (28+0-3)/1+1
->  32, 13, 13 -> 64, 11, 11          11 = (13+0-3)/1+1
->  64, 5, 5                           5 = int ((11-2)/2+1)
->  64*5*5=1600
"""

#将模型转移到GPU中
model = Model().to(device)

#设置hyperpatameter
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)             # 训练集的大小，一共60000张图片
    num_batches = len(dataloader)              # 批次数目，1875（60000/32）
    train_loss , train_acc = 0,0
    for X, y in dataloader:  # 60000/32
        X, y = X.to(device), y.to(device)

        # 计算预测误差
        pred = model(X)                       # 网络输出
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()

    train_loss /= num_batches
    train_acc /= size

    return train_acc, train_loss

def test(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss , test_acc = 0, 0

    # 当不进行训练时，停止梯度更新，节省计算内存消耗

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            test_loss += loss.item()
            test_acc += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_acc /= size
    test_loss /= num_batches
    return test_acc, test_loss

"""
1. pred.argmax(1) 返回数组 pred 在第一个轴（即行）上最大值所在的索引。这通常用于多类分类问题中，其中 pred 是一个包含预测概率的二维数组，每行表示一个样本的预测概率分布
2. (pred.argmax(1) == y)是一个布尔值，其中等号是否成立代表对应样本的预测是否正确（True 表示正确，False 表示错误）
3. .type(torch.float)是将布尔数组的数据类型转换为浮点数类型，即将 True 转换为 1.0，将 False 转换为 0.0。
4. .sum()是对数组中的元素求和，计算出预测正确的样本数量。
5. .item()将求和结果转换为标量值，以便在 Python 中使用或打印。
6. pred 是一个二维数组，通常是一个 预测概率矩阵。每一行表示一个样本的预测概率分布，每一列对应一个类别的预测概率。
"""

epochs = 5
train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    model.train()
    epoch_train_loss, epoch_train_acc = train(train_dl, model, loss_fn, optimizer)
    train_loss.append(epoch_train_loss)
    train_acc.append(epoch_train_acc)
    model.eval()
    # 这里是为什么？
    epoch_test_loss, epoch_test_acc = test(test_dl, model, loss_fn, optimizer)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

    print("Epoch:{:2d}, Train_acc:{:.1f}%, Train_loss:{:.3f}, Test_acc:{:.1f}%, Test_loss:{:.3f}".format(epoch + 1,
                                        epoch_train_acc * 100, epoch_train_loss, epoch_test_acc * 100, epoch_test_loss))
print('Done')


import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']    = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号
plt.rcParams['figure.dpi']         = 100        #分辨率

epochs_range = range(epochs)

plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)

plt.plot(epochs_range, train_acc, label='Training Accuracy')
plt.plot(epochs_range, test_acc, label='Test Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')   # 打卡请带上时间戳，否则代码截图无效

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, test_loss, label='Test Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()