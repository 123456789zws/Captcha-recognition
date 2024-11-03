import os
import torch
from Modules.Model import VerModel
from torchvision import transforms
from PIL import Image
from Modules.utils.FixSplit import fixSp


# 进行识别
def modelVerify(in_tensor, model):
    out_tensor = model(in_tensor)
    return out_tensor


def identify(dir, device='cpu'):
    transformToTensor = transforms.ToTensor()
    transformResize = transforms.Resize((32, 32))
    # 实例化模型
    model = VerModel(in_channels=3, num_class=10).to(device)
    model.load_state_dict(torch.load('Modules/weights/VerModel.pth'))
    # 进行分割
    img = Image.open(dir)
    imgTensor = transformToTensor(img)
    img_block_ls = fixSp(imgTensor)
    res = ""
    for img_block in img_block_ls:
        prob = modelVerify(transformResize(img_block).unsqueeze(0), model)
        ans = torch.max(prob, dim=1)[1]
        res += str(ans.item())
    return res


def main(imgDir):
    res = identify(imgDir)
    return res


if __name__ == '__main__':
    expDir = r'./ExperimentalData'
    filesName = os.listdir(expDir)
    # 进行拼接
    filesPaths = [os.path.join(expDir, name) for name in filesName]
    # 对每一个图片分别进行预测
    for path in filesPaths:
        res = main(path)
        print(res)