# from ultralytics import YOLO
#
# if __name__=='__main__':
#     model = YOLO('./yolo11x-cls.pt')
#
#     results = model.train(data='rock', epochs=100, imgsz=64)

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x.yaml")  # build a new model from YAML
model = YOLO("yolo11x.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11x.yaml").load("yolo11x.pt")  # build from YAML and transfer weights

# Train the model
# results = model.train(data="mydata.yaml", epochs=100, imgsz=640)
results = model.train(data="mydata.yaml", epochs=100, imgsz=640, batch=16)