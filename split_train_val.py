import os
import random
import shutil

folder_path=r'D:\target_detection\ultralytics-main\sedimentary_rock'
class_=os.listdir(folder_path)

for item in class_:
    src_folder=os.path.join(folder_path,item)
    train_folder = os.path.join(r'D:\target_detection\ultralytics-main\sedimentary_rock1\test',item)  # 训练集目标路径
    test_folder = os.path.join(r'D:\target_detection\ultralytics-main\sedimentary_rock1\val',item)    # 测试集目标路径
    os.mkdir(train_folder)
    os.mkdir(test_folder)
    # 划分比例（训练集:测试集 = 8:2）
    split_rate = 0.8
    # 获取原文件路径
    images = os.listdir(src_folder)

    # 获取原文件下图片的数量
    num = len(images)

    # ------------------随机采样训练集的索引------------------------
    train_index = random.sample(images, k=int(num * split_rate))
    # ------------------不使用 随机采样----------------------------
    # k=int(num * split_rate)
    # train_index = images[:k]
    # -----------------------------------------------------------

    for index, image in enumerate(images):
        if image in train_index:
            # 将原文件路径的图片移动到训练集路径下
            image_path = os.path.join(src_folder, image)
            shutil.move(image_path, train_folder)
        else:
            # 将原文件路径的图片移动到测试集路径下
            image_path = os.path.join(src_folder, image)
            new_path = os.path.join(image_path, test_folder)
            shutil.move(image_path, new_path)