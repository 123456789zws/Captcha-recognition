from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from Modules.utils.FixSplit import fixSp      # 使用绝对导入

transform = transforms.Compose([transforms.ToTensor()])
transformToPil = transforms.ToPILImage()
transformResize = transforms.Resize((32, 32))


#进行显示数据
def showImage(in_tensor):
    img_tensor = in_tensor.permute(1, 2, 0).numpy()
    plt.imshow(img_tensor)
    plt.show()
    plt.close()

# 读取数据
def ReadImage(oDir):
    oFilesName = os.listdir(oDir)
    oFiles = [os.path.join(oDir, f) for f in oFilesName]
    return oFiles


# 分割数据
def SplitImage(imgDir):
    img = Image.open(imgDir)
    imgTensor = transform(img)
    # shape = (3, 50, 135)
    img_ls = fixSp(imgTensor)
    # Resize = (3, 32, 32)
    resizeImg = []
    for img_ in img_ls:
        img_resize = transformResize(img_)
        resizeImg.append(img_resize)
    return img_ls


# 保存数据
def SaveSplitImage(img_tensor, pFile: str):
    # 进行分割路径
    dir, _ = os.path.split(pFile)
    # 检查目标文件是否存在，不存在则创建
    if os.path.exists(dir) == False:
        os.makedirs(dir)

    img = transformToPil(img_tensor)
    img.save(pFile)


# 根据参数对训练集和测试集进行操作
def TrainOrTest(dirFiles, tarFiles):
    for idx, otf in enumerate(dirFiles):
        img_1, img_2, img_3, img_4 = SplitImage(otf)
        _, name = os.path.split(otf)    # 提取文件名称
        _, suffix = os.path.splitext(name)  # 提取文件后缀
        name_1, name_2, name_3, name_4 = name[:4]
        SaveSplitImage(img_1, os.path.join(tarFiles, name_1, f'{idx}_1'+suffix))
        SaveSplitImage(img_2, os.path.join(tarFiles, name_2, f'{idx}_2'+suffix))
        SaveSplitImage(img_3, os.path.join(tarFiles, name_3, f'{idx}_3'+suffix))
        SaveSplitImage(img_4, os.path.join(tarFiles, name_4, f'{idx}_4'+suffix))
def main():
    # 获取训练集与测试集文件路径
    oTrainDir = os.path.join(OROOT_DIR, 'Train')
    oTestDir = os.path.join(OROOT_DIR, 'Test')
    oTrainFiles = ReadImage(oTrainDir)
    oTestFiles = ReadImage(oTestDir)

    # 目标训练集与测试集路径
    pTrainDir = os.path.join(TROOT_DIR, 'Train')
    pTestDir = os.path.join(TROOT_DIR, 'Test')

    if os.path.exists(pTrainDir) == False:
        os.makedirs(pTrainDir)

    if os.path.exists(pTestDir) == False:
        os.makedirs(pTestDir)

    # 首先对训练集进行分割
    TrainOrTest(oTrainFiles, pTrainDir)

    # 对测试集进行分割
    TrainOrTest(oTestFiles, pTestDir)


if __name__ == '__main__':
    OROOT_DIR = 'MDataSet'
    TROOT_DIR = 'PDataSet'
    main()
