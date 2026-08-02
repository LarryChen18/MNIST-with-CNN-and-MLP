import torch
def test(model, test_loader,device):
    model.eval()  # 设为评估模式（禁用Dropout,batch_norm等）
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)#1表示取每行（每张图片）中的最大值，_是占位符，表示不取最大值本身而是取索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()#item将SUM后的单个值转化为整形，tenser变为int
    # 打印测试结果
    accuracy = 100 * correct / total
    return accuracy