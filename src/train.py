import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from models.CNN import CNN
from models.MLP import MLP
from config import load_config
from test import test
from utils import plot_loss
from utils import plot_accuracy
from utils import save_model
from utils import demo
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()#告诉pytorch进入训练模式，将Dropout、BatchNorm等层切换到训练状态。
    running_loss = 0.0
    for images, labels in train_loader:
        # 将数据移动到GPU
        images = images.to(device)
        labels = labels.to(device)#形状为(batch_size,),记录实际的数字
        # 前向传播
        outputs = model(images)
        # outputs形状为(batch_size,10),每一行代表0-9十个数字的预测结果,此时并未转化为概率，它将在loss计算时通过softmax计算
        # 计算loss
        loss = criterion(outputs, labels)
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss
def main():
    # 读取配置文件
    config = load_config()
    # 判断使用GPU还是CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载数据
    batch_size = config["dataset"]["batch_size"]
    train_loader, test_loader = get_dataloader(batch_size)
    # 根据配置选择模型
    model_name = config["model"]["name"]
    if model_name == "CNN":
        model = CNN()
    elif model_name == "MLP":
        model = MLP()
    else:
        raise ValueError("Unknown model name")#防止config文件输入错误而导致model没有值而报错
    # 将模型移动到设备
    model = model.to(device)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()#交叉熵损失
    # 定义优化器
    lr = config["train"]["lr"]
    optimizer = optim.Adam(model.parameters(),lr=lr)
    # 保存每轮loss
    loss_history = []
    acc_history = []
    best_accuracy = 0
    epochs = config["train"]["epochs"]
    # 正式训练
    for epoch in range(epochs):
        loss = train_one_epoch(model,train_loader,criterion,optimizer,device)
        acc = test(model,test_loader,device)
        loss_history.append(loss)
        acc_history.append(acc)
        if acc > best_accuracy:
            best_accuracy = acc
            save_model(model,'checkpoints/best_model.pth')
        print(f"Epoch [{epoch+1}/{epochs}], Loss:{loss:.4f}, Accuracy:{acc:.2f}%, Best-Accuracy:{best_accuracy:.2f}%")
    # 绘制loss曲线
    plot_loss(loss_history,"results/loss.png")
    # 绘制accuracy曲线
    plot_accuracy(acc_history,"results/accuracy.png")
    print("Training Finished")
    demo(model,device,test_loader,"results/demo.png")
if __name__ == "__main__":
    main()