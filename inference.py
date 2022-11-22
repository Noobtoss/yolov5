import torch
import cv2
# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='results/bestX.pt')  # local model
model.conf = 0.5
# Image
img = 'manuel_all_1.jpg'

# Inference
results = model(img)
results.show()
print(results)
print(results.pandas().xyxy[0])
#cv2.imwrite('manuel_all_1.jpg', results)