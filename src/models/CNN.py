import torch
import torch.nn as nn 
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 5,1,2),#输入（只有一个灰度通道），输出通道数（特征图数）：意味着有32个卷积核，用于从输入图片中提取32个特征
                                   # 卷积核大小，步长，边缘填充层数
                                   nn.ReLU(),#作非线性筛选，正数不变，负数变0，删除无关紧要的特征，保留重要的得分高的特征
                                   nn.MaxPool2d(2))#最大池化层：把特征图分成2*2的网格，每个网格四个激活值，保留最大的激活值（最能体现该特征图的特征，则得到的激活值y最大），图片大小缩小到1/4。
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5,1,2),#（28-5+2*2)/1+1=28。。在32个低级特征的基础上，再提取64个高级特征，更细致的描述特征
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))
        self.out= nn.Linear(64*7*7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)#或者(-1,64*7*7)
        x = self.out(x)
        return x