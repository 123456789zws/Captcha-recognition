import torch
from Model import VerModel
from torchvision import transforms
from PIL import Image
from utils.FixSplit import fixSp


# 进行识别
def modelVerify(in_tensor, model):
    out_tensor = model(in_tensor)
    return out_tensor

def identify(dir, device='cpu'):
    transformToTensor = transforms.ToTensor()
    transformResize = transforms.Resize((32, 32))
    # 实例化模型
    model = VerModel(in_channels=3, num_class=10).to(device)
    model.load_state_dict(torch.load('weights/VerModel.pth'))
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


if __name__ == '__main__':
    res = identify('DataProce/MDataSet/Test/3758.png')
    print(res)