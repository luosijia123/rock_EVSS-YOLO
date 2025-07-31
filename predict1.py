# from ultralytics import YOLO
#
# # Load a model
# model = YOLO("yolo11x-cls.pt")  # load an official model
# model = YOLO("best.pt")  # load a custom model
#
# # Predict with the model
# results = model("amphibolite0.jpg")  # predict on an image
# print("---------------------------")
# print(results[0].probs.top1)
#

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load an official model
model = YOLO("best.pt")  # load a custom model

# Predict with the model
result = model(r"C:\Users\35792\Desktop\3.jpeg",save=True)  # predict on an image

# print(results[0].boxes)
# print(float(results[0].boxes.xywh[0][0]))