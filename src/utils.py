import os
import matplotlib.pyplot as plt
import torch
def create_dir(path):#如果没有对应名称的文件夹则自动创建
    if not os.path.exists(path):
        os.makedirs(path)
def plot_loss(history, save_path):#绘制损失与训练轮次的图像
    create_dir(os.path.dirname(save_path))
    plt.figure()
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(save_path)#保存图片到指定路径
    plt.close()
def plot_accuracy(history, save_path):#绘制准确率与训练轮次的图像
    create_dir(os.path.dirname(save_path))
    plt.figure()
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy")
    plt.savefig(save_path)
    plt.close()
def save_model(model, save_path):
    create_dir(os.path.dirname(save_path))
    torch.save(model.state_dict(),save_path)
def demo(model,device,test_loader,save_path):
    create_dir(os.path.dirname(save_path))
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        _, pred = torch.max(output, 1)
        plt.figure(figsize=(15, 6))
        for i in range(10):
            plt.subplot(2, 5, i + 1)  # 两行五列排列10张子图
            #plt.imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)), cmap='gray')  # 要放cpu上，plt不支持gpu
            plt.imshow(images[i].cpu().squeeze(0),cmap='gray')#既支持tenser也支持numpy
            plt.title(f'true:{labels[i].item()},pred:{pred[i].item()}')
            plt.axis('off')
        plt.savefig(save_path)
        plt.show()
