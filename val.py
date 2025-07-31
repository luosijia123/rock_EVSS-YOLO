from ultralytics import YOLO

# model=YOLO()
#
# metrics = model.val()
# metrics.top1
# metrics.top5

model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')

results = model.val(data='mydata.yaml',imgsz=640)