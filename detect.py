from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = YOLO('best(1).pt')
results = model('test_data/sandstone26.jpg',save=True,visualize=True)