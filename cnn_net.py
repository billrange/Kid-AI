import sys, os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets,  transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision.io import read_image
import matplotlib.pyplot as plt
from PIL import Image
# import cv2
import timeit
import pandas as pd

LABEL_NUM = 2 # 识别种类个数
FEATURE_LEN = 224*224*3 # 图像Flattened 后的输入神经网络特征的个数
startTime = timeit.default_timer() #记录程序运行起始时间
batch_size = 64

# ----- 图像数据格式化
data_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)), # 让（图像高, 图像宽）=(224, 224)
    transforms.ToTensor(),
])

# -----任务1： 通过不同的文件夹来导入数据
# 数据集格式:
#          root/
#            ├── train
#            │   ├── Dog
#            │   │    ├── ...
#            │   │    └── *.jpg
#            │   └── Cat
#            │        ├── ...
#            │        └── *.jpg
#            └── test
#                ├── Dog
#                │    ├── ...
#                │    └── *.jpg
#                └── Cat
#                     ├── ...
#                     └── *.jpg

root = "/home/bill/Downloads/Cat_Dog_data_subset/" #数据集上一层次目录
dataset_train = ImageFolder(root + "train/",transform = data_transform) # 导入train图像数据集
train_dataloader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True) #制作train数据集

# ----- 任务2： 通过CSV来读取图像位置信息和类别信息
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):

        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# 需要使用data_transform, 可以把测试数据变为和训练数据一样的类型
dataset_test = CustomImageDataset(annotations_file=root+"test/"+"sampleSubmission.csv",
    img_dir=root+"test/",transform = data_transform) # 导入test图像数据集
test_dataloader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True) #制作test数据集

# ----- 显示猫狗图像
for X, y in test_dataloader:
    print('      ----- 1. 查看数据集的形状 -----')
    print(" - 数据 X 的形状[数据个数, 颜色通道数, 图像高, 图像宽]: ", X.cpu().numpy().shape)
    print(" - 标签 y 的长度: ", len(y))
    print()
    break
# ----- 随机显示猫狗数据集
labels_map = {
    0: "Cat",
    1: "Dog",
}

figure = plt.figure(figsize=(8, 32))
cols, rows = 3, 2
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(dataset_train), size=(1,)).item() # 产生随机样本编号
    img, label = dataset_train[sample_idx] # 获得指定样本编号下的图像和对应的标签
    figure.add_subplot(rows, cols, i) # 让计算机留出 3 × 2的绘图空间
    plt.title(labels_map[label]) # 在图像上显示标签
    plt.axis("off") # 不要显示图像坐标
    plt.imshow(img.detach().cpu().squeeze().numpy().transpose(1,2,0)) # 在画布上绘制图像
plt.show() # 显示猫狗图像

# ----- 查看使用的硬件设备.
device = "cuda" if torch.cuda.is_available() else "cpu"
print('      ----- 2. 查看使用的硬件设备运行神经网络模型 -----')
print(" - 使用 {} 设备。\n".format(device))
# ----- 任务3： 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(FEATURE_LEN, 512), # 数据线性变换 FEATURE_LEN （个）--> 512 （个）
            nn.ReLU(), # 数据非线性变换， 小于0的让其为0
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, LABEL_NUM), # 最后需要把数据变为训练种类的个数
            nn.ReLU()
        )

    def forward(self, x): # 把图像x输入到上面构建的神经网络中，进行向前推理
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device) # 从抽象到具体构建神经网络模型
print('      ----- 3. 使用的神经网络模型结构 -----')
print(model,'\n')

loss_fn = nn.CrossEntropyLoss() # 定义损失函数，即了解预测的猫狗种类与真实的偏差
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # 定义梯度下降算法，即尽量寻找上面损失值最小点

# ----- 任务4.1： 建立训练模型
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) # 数据长度
    for batch, (X, y) in enumerate(dataloader): # 把数据分为不同batch进行训练
        correct = 0
        X, y = X.to(device), y.to(device)

        # 计算预测误差
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #显示训练效果
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            correct /= len(X)
            print(f"损失值: {loss:>7f}  [{current:>5d}/{size:>5d}], 精度: {(100*correct):>0.1f}% ")

# ----- 任务4.2： 建立测试模型
def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f" - 预测结果: \n   精度: {(100*correct):>0.1f}%,   平均损失值: {test_loss:>8f} \n")

# ----- 任务4.2： 训练和测试模型在猫狗图像分类中的效果
print('      ----- 4. 人工神经网络开始训练和预测猫狗数据集 -----')
epochs = 5 #迭代5次
for t in range(epochs):
    print(f"迭代次数： {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer) # 训练
    test(test_dataloader, model) # 预测

stopTime = timeit.default_timer() #记录停止时间
print('完成了! 程序全部运行耗时: %5.1fs。'%(stopTime - startTime))
