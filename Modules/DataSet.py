import os

import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        '''
        :param root_dir: 表示训练集路径或测试集路径
        :param image_dir:
        :param transform:
        '''
        self.all_dir = root_dir
        self.label_n = os.listdir(root_dir)     # 所有标签名称
        self.image_dir = [os.path.join(root_dir, nd) for nd in self.label_n]    # 得到所有数字文件的路径
        self.transform = transform
        self.labels, self.images= [], []
        # 得到每一个文件夹下的img
        for idx, fle in enumerate(self.image_dir):
            number_files = [os.path.join(fle, f) for f in os.listdir(fle)]  # 数字目录下的所有文件
            t_labels = [int(self.label_n[idx]) for _ in range(len(number_files))]
            self.images.extend(number_files)
            self.labels.extend(t_labels)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 得到每一个图片的路径
        img_name = self.images[idx]
        image = Image.open(img_name).convert('RGB')  # 读取图像并转换为RGB格式
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
            label = torch.tensor(label)

        return image, label


if __name__ == '__main__':
    ROOTDIR = "DataProce/PDataSet/Train"
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImageDataset(ROOTDIR)
    image, label = dataset[-1]
    image.show()
    print(label)
    print(len(dataset))