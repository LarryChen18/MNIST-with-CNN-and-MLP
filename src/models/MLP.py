import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 将28x28的图像展平为784维向量，-1是自动计算的通配符，自动计算出满足784维的情况,即得到(batch_size,28*28)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # 输出层不需要激活，因为后续用CrossEntropyLoss（包含Softmax）
        return x