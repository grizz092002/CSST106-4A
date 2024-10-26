# YOLOv3 Object Detection

This repository demonstrates how to use the YOLOv3 (You Only Look Once) model for real-time object detection on images. YOLO is a fast and accurate object detection algorithm, which performs detection in a single pass, making it suitable for applications where real-time performance is essential.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Code Overview](#code-overview)
   - [Model Loading](#1-model-loading)
   - [Image Input](#2-image-input)
   - [Object Detection](#3-object-detection)
   - [Non-Maximum Suppression](#4-non-maximum-suppression)
   - [Visualization](#5-visualization)
   - [Testing Multiple Images](#6-testing-multiple-images)
4. [Performance Analysis](#performance-analysis)

## Prerequisites

1. **Python Libraries**:
   - `cv2` (OpenCV)
   - `numpy`
   - `matplotlib`

2. **Model Files**:
   - [YOLOv3 Configuration (`yolov3.cfg`)](https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg)
   - [YOLOv3 Weights (`yolov3.weights`)](https://pjreddie.com/media/files/yolov3.weights)

3. **Image Files**:
   - Ensure you have the images you want to test. Here, we refer to `/content/image.jpg`, `/content/test1.jpg`, `/content/test2.jpg`, and `/content/test3.jpg`.

## Installation

Download the YOLOv3 configuration and weights files:

```bash
!wget https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg
!wget https://pjreddie.com/media/files/yolov3.weights
```
## Code Overview
   ## 1. Model Loading
   ```bash
   import cv2
   import numpy as np
   import matplotlib.pyplot as plt
   
   # Load the YOLO model
   net = cv2.dnn.readNet("/content/yolov3.weights", "/content/yolov3.cfg")
   
   # Get the output layer names
   layer_names = net.getLayerNames()
   output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
   ```
   - Load the pre-trained YOLO model with weights and configuration.
   - Retrieve the output layer names for prediction.
     
   ## 2. Image Input
   ```bash
   # Load the image
   image_path = '/content/image.jpg'
   image = cv2.imread(image_path)
   height, width, channels = image.shape
   
   # Prepare the image for YOLO
   blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
   net.setInput(blob)
   ```
   - Read the image and preprocess it to the required format for YOLO: resize, normalize, and set it as input.
   
   ## 3. Object Detection
   ```bash
   # Perform the forward pass
   detections = net.forward(output_layers)
   
   # Define lists to hold the information
   class_ids = []
   confidences = []
   boxes = []
   
   # Loop over each of the detections
   for output in detections:
       for detection in output:
           scores = detection[5:]
           class_id = np.argmax(scores)
           confidence = scores[class_id]
   
           if confidence > 0.5:
               center_x = int(detection[0] * width)
               center_y = int(detection[1] * height)
               w = int(detection[2] * width)
               h = int(detection[3] * height)
   
               x = int(center_x - w / 2)
               y = int(center_y - h / 2)
   
               boxes.append([x, y, w, h])
               confidences.append(float(confidence))
               class_ids.append(class_id)
   ```
   - Perform a forward pass through the model to get the detections.
   - Parse detection outputs and store bounding boxes, class IDs, and confidence scores.

   ## 4. Non-Maximum Suppression
   ```bash
   indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
   ```
   - Apply NMS to remove redundant overlapping bounding boxes, keeping only the box with the highest confidence for each object.

   ## 5. Visualization
   ```bash
   import matplotlib.pyplot as plt
   import numpy as np
   
   # Create a list of colors for the bounding boxes
   colors = np.random.uniform(0, 255, size=(len(boxes), 3))
   
   # Draw the bounding boxes and labels on the image
   if len(indices) > 0:
       for i in indices.flatten():
           x, y, w, h = boxes[i]
           label = str(class_ids[i])
           confidence = confidences[i]
           color = colors[i]
   
           cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
           cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
   
   # Display the image with detected objects
   plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
   plt.show()
   ```
   ![4A-COLLANTES-MP5-Process_Image4](https://github.com/user-attachments/assets/0d804ab7-ed41-45a4-9e28-b7260f230807)
   
   - Draw bounding boxes and confidence scores on the image for each detected object.
   - Display the processed image using Matplotlib.
