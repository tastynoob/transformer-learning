import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import numpy as np

import mnist


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.linear(x)
        out = self.relu(out)
        return out + residual


class BPNetWork(nn.Module):
    def __init__(self):
        super(BPNetWork, self).__init__()
        self.layer1 = nn.Linear(784, 64)
        self.relu1 = nn.ReLU()
        self.residual = ResidualBlock(64, 64)
        self.layer4 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.residual(x)
        x = self.layer4(x)
        return F.softmax(x, dim=1)


def absolute_loss(y_true, y_pred):
    """自定义绝对值损失函数"""
    return torch.sum(torch.abs(y_true - y_pred))


def prepare_data():
    """准备数据集"""
    train_data = mnist.getTrainData()
    test_data = mnist.getTestData()
    
    # 转换训练数据
    train_images = []
    train_labels = []
    for img, label_index in train_data:
        train_images.append(img.flatten())
        # 创建one-hot编码
        label = np.zeros(10)
        label[label_index] = 1
        train_labels.append(label)
    
    # 转换测试数据
    test_images = []
    test_labels = []
    for img, label_index in test_data:
        test_images.append(img.flatten())
        label = np.zeros(10)
        label[label_index] = 1
        test_labels.append(label)
    
    # 转换为PyTorch张量
    train_images = torch.FloatTensor(np.array(train_images))
    train_labels = torch.FloatTensor(np.array(train_labels))
    test_images = torch.FloatTensor(np.array(test_images))
    test_labels = torch.FloatTensor(np.array(test_labels))
    
    return train_images, train_labels, test_images, test_labels


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = BPNetWork().to(device)
    
    # 准备数据
    train_images, train_labels, test_images, test_labels = prepare_data()
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)
    
    print("Dataset loaded successfully")
    
    # 创建数据加载器
    train_dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    # 使用普通梯度下降算法
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    
    # 训练循环
    for epoch in range(20):
        print(f"Epoch {epoch + 1}")
        total_loss = 0
        model.train()
        
        for batch_idx, (data, target) in enumerate(tqdm.tqdm(train_loader)):
            optimizer.zero_grad()
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            loss = absolute_loss(target, output)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Total loss: {total_loss / len(train_loader):.4f}")
    
    # 测试模型
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(test_images))):
            img = test_images[i:i+1]
            label_index = torch.argmax(test_labels[i]).item()
            
            output = model(img)
            predicted = torch.argmax(output).item()
            
            total += 1
            if predicted == label_index:
                correct += 1
    
    print(f"Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()