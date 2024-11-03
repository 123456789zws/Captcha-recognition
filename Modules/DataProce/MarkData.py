# 对所有图片进行标注
from PIL import Image
import matplotlib.pyplot as plt
import time
import os


# 进行存储
def re_save(name, image, n_dir):
    whole_dir = os.path.join(n_dir, name)
    image.save(whole_dir + '.png')


def show_pic(dir: str):
    image = Image.open(dir)
    plt.imshow(image)
    plt.show(block=True)
    # plt.pause(3)
    plt.close()
    return image


def main(oRootDir: str, nRootDir: str):
    TrainDir = os.path.join(oRootDir, 'Train')
    TestDir = os.path.join(oRootDir, 'Test')
    Train_files_ls = os.listdir(TrainDir)
    Test_files_ls = os.listdir(TestDir)
    # 将每一个数据都拼接为完整的相对路径
    Train_files = [os.path.join(TrainDir, nf) for nf in Train_files_ls]
    Test_files = [os.path.join(TestDir, tf) for tf in Test_files_ls]
    # print(Train_files)
    nTrain_files = os.path.join(nRootDir, "Train")
    nTest_files = os.path.join(nRootDir, "Test")
    if os.path.exists(nTrain_files) == False:
        os.makedirs(nTrain_files)
    if os.path.exists(nTest_files) == False:
        os.mkdir(nTest_files)

    # 设定新的路径

    # 进行训练集标注
    for idx, trainf in enumerate(Train_files):
        if idx <= SCHEDULE:
            continue
        # 1. 进行显示
        img = show_pic(trainf)
        # 2. 进行输入
        calib = str(input(f"《训练集》当前进度：【{idx} / {len(Train_files)}】，请输入你看到的数字："))
        while len(calib) != 4 or calib == 'n':
            if calib == 'n':
                _ = show_pic(trainf)
                calib = str(input(f"《训练集》当前进度：【{idx} / {len(Train_files)}】，请输入你看到的数字："))
            else:
                calib = str(input("ERROR, 请重新输入："))

        # 3. 进行存储
        re_save(calib, img, nTrain_files)
        if idx % 10 == 0:
            with open('schedule.txt', 'a+', encoding='utf-8') as file:
                file.write(f'Train: {idx}\t')
    # 进行测试集标注
    for idx, testf in enumerate(Test_files):

        # 1. 进行显示
        img = show_pic(testf)
        # 2. 进行输入
        calib = input(f"《测试集》当前进度：【{idx} / {len(Test_files)}】，请输入你看到的数字：")
        while len(calib) != 4 or calib == 'n':
            if calib == 'n':
                _ = show_pic(testf)
                calib = input(f"《测试集》当前进度：【{idx} / {len(Test_files)}】，请输入你看到的数字：")
            else:
                calib = str(input("ERROR, 请重新输入："))
        # 3. 进行存储
        # 3. 进行存储
        re_save(calib, img, nTest_files)
        if idx % 10 == 0:
            with open('schedule.txt', 'a+', encoding='utf-8') as file:
                file.write(f'Test: {idx}\t')


if __name__ == "__main__":
    O_ROOTDIR = "Modules\\DataProce\\DataSet"
    N_ROOTDIR = "Modules\\DataProce\\MDataSet"
    SCHEDULE = 810
    main(O_ROOTDIR, N_ROOTDIR)
