import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from DataSet import ImageDataset
from Model import VerModel
from tqdm import tqdm
import argparse


def main(args):
    # 取出数据
    epochs, device, batch_size, save_path = args.epochs, args.device, args.batch_size, args.save_path

    # 定义转换
    transform = transforms.Compose([transforms.ToTensor()])
    # 加载训练集和测试集
    train_dataset = ImageDataset('Modules/DataProce/PDataSet/Train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = ImageDataset('Modules/DataProce/PDataSet/Test', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 定义模型、损失函数、优化器
    model = VerModel().to(device)
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    TXT = []
    # 进行训练
    for epoch in tqdm(range(epochs), desc="【ALL Epochs】", position=0):
        model.train()
        # 训练
        running_loss = 0.0
        for inputs, labels in tqdm(train_dataloader, desc='<Train Schedule>', position=1, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lossFunction(outputs, labels.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'《Epoch》: [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_dataloader)}')
        TXT.append(f'\t《Epoch》: [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_dataloader)} \n')

        # 测试
        if (epoch + 1) % 5 == 0:     # 设置每间隔5次检验一次效果
            model.eval()
            accuracy = 0.0
            with torch.no_grad():
                for inputs, labels in tqdm(test_dataloader, desc='<Test Schedule>', position=2, leave=False):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    predict = torch.max(outputs, dim=1)[1]
                    acc_nums = torch.eq(predict, labels).sum().item()
                    accuracy += acc_nums
                print("《Accuracy》: ", accuracy / len(test_dataset))
                TXT.append("\t《Accuracy》: {} \n\n".format(accuracy / len(test_dataset)))

    # 进行保存数据
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, 'VerModel.pth'))
    with open('weights/Record.txt', 'w+', encoding='utf-8') as file:
        file.writelines(TXT)


if __name__ == '__main__':
    # 创建解析器
    parser = argparse.ArgumentParser(description="Greet someone with a message.")
    # --epochs 是一个可选参数（选择性的从命令行传参，具有default）， epochs 是一个位置参数（即：必须由命令行传参，没有default）
    parser.add_argument("--epochs", type=int, default=50, help="循环迭代的次数")
    parser.add_argument("--device", type=str, default='cpu', help="程序运行的设备")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument('--save-path', type=str, default='weights', help='权重保存的路径')
    args = parser.parse_args()
    main(args)