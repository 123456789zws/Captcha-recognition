# 对所有图片进行标注
from PIL import Image
import matplotlib.pyplot as plt
import time
import os


# 进行存储
def rename_file(old_name,new_name):
    os.rename(old_name, new_name)

def show_pic(dir: str):
    _, name = os.path.split(dir)
    image = Image.open(dir)
    plt.title(name[:-4], fontsize=80)
    plt.imshow(image)
    plt.show(block=True)
    # plt.pause(3)
    plt.close
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
        if idx == 0:
            break
        if idx % 10 == 0:
            with open('schedule_v.txt', 'a+', encoding='utf-8') as file:
                file.write(f'Train: {idx}\t')
        # 1. 进行显示
        img = show_pic(trainf)
        # 2. 进行输入
        calib = str(input(f"《训练集》当前进度：【{idx} / {len(Train_files)}】，请输入你看到的数字："))
        if calib == "":
            continue
        new_name = os.path.join(TrainDir, calib+'.png')
        if os.path.exists(new_name):
            new_name = os.path.join(TrainDir, calib+'-c.png')
        # 重命名
        rename_file(trainf, new_name)

    # 进行测试集标注
    for idx, testf in enumerate(Test_files):
        if idx % 10 == 0:
            with open('schedule_v.txt', 'a+', encoding='utf-8') as file:
                file.write(f'Test: {idx}\t')

        if idx < SCHEDULE:
            continue
        # 1. 进行显示
        img = show_pic(testf)
        # 2. 进行输入
        calib = input(f"《测试集》当前进度：【{idx} / {len(Test_files)}】，请输入你看到的数字：")
        if calib == "":
            continue
        new_name = os.path.join(TestDir, calib+'.png')
        if os.path.exists(new_name):
            new_name = os.path.join(TestDir, calib+'-c.png')
        # 重命名
        rename_file(testf, new_name)




if __name__ == "__main__":
    N_ROOTDIR = "Modules\\DataProce\\MDataSet"
    SCHEDULE = 418
    main(N_ROOTDIR, N_ROOTDIR)


