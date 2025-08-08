import os
import random
import shutil

folder_path=r'your_path'
class_=os.listdir(folder_path)

for item in class_:
    src_folder=os.path.join(folder_path,item)
    train_folder = os.path.join(r'your_path\train',item)
    test_folder = os.path.join(r'your_path\val',item)
    os.mkdir(train_folder)
    os.mkdir(test_folder)
    split_rate = 0.7
    images = os.listdir(src_folder)

    num = len(images)

    train_index = random.sample(images, k=int(num * split_rate))

    for index, image in enumerate(images):
        if image in train_index:
            image_path = os.path.join(src_folder, image)
            shutil.move(image_path, train_folder)
        else:
            image_path = os.path.join(src_folder, image)
            new_path = os.path.join(image_path, test_folder)
            shutil.move(image_path, new_path)