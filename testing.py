from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-face.yaml")  # build a new model from scratch
model = YOLO("yolov8n-face.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="coco128.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("C:\\xampp1\\htdocs\\demo\\face_recogition_intel\\face_img\\HR004-8.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format