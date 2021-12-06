import cv2
import matplotlib.pyplot as plt
import os
import datetime
from tensorflow.keras.preprocessing import image
import matplotlib.image as mpimg
import numpy as np
from tensorflow import keras
import joblib



# Load model YOLO
MODEL = 'Yolo\yolov3-face.cfg'
WEIGHT = 'Yolo\yolov3-wider_16000.weights'
net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
label_dict={0: 'Ngoc', 1: 'Yen', 2: 'Duc_Anh', 3: 'Hang', 4: 'Hoi'}
IMG_WIDTH, IMG_HEIGHT = 416, 416
knn_from_joblib = joblib.load('Model\Result\knn.pkl') 
def load_and_preprocess(frame):
    img = cv2.resize(frame, (300, 300), interpolation = cv2.INTER_AREA)
    img = img / 255.
    img = img.flatten()
    return img

def get_predict_yolo(frame):

    # Making blob object from original image
    blob = cv2.dnn.blobFromImage(frame, 
                                1/255, (IMG_WIDTH, IMG_HEIGHT),
                                [0, 0, 0], 1, crop=False)

    # Set model input
    net.setInput(blob)

    # Define the layers that we want to get the outputs from
    output_layers = net.getUnconnectedOutLayersNames()

    # Run 'prediction'
    outs = net.forward(output_layers)

    blobb = blob.reshape(blob.shape[2] * blob.shape[1], blob.shape[3]) 
    print(blobb.shape)

    # plt.figure(figsize=(15, 15))
    # plt.imshow(blobb, cmap='gray')
    # plt.axis('off')
    # plt.show()
    return outs

def threshold_boxes(outs):
  frame_height = frame.shape[0]
  frame_width = frame.shape[1]

# Scan through all the bounding boxes output from the network and keep only
# the ones with high confidence scores. Assign the box's class label as the
# class with the highest score.
  confidences = []
  boxes = []
  # Each frame produces 3 outs corresponding to 3 output layers
  for out in outs:
  # One out has multiple predictions for multiple captured objects.
      for detection in out:
        confidence = detection[-1]
        # Extract position data of face area (only area with high confidence)
        if confidence > 0.5:
            center_x = int(detection[0] * frame_width)
            center_y = int(detection[1] * frame_height)
            width = int(detection[2] * frame_width)
            height = int(detection[3] * frame_height)
            
            # Find the top left point of the bounding box 
            topleft_x = int(center_x - width/2)
            topleft_y = int(center_y - height/2)
            confidences.append(float(confidence))
            boxes.append([topleft_x, topleft_y, width, height])
  return boxes,confidences

def nms_boxes(boxes,confidences):
        # Perform non-maximum suppression to eliminate 
    # redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    final_boxes=[]
    result = frame.copy()
    crop=frame.copy()
    for i in indices:
        i = i[0]
        box = boxes[i]
        final_boxes.append(box)

        # Extract position data
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        red = (0,0,255)
        # Draw bouding box with the above measurements
        ### YOUR CODE HERE
        cv2.rectangle(result,(left,top)  ,(left + width,top + height),color=red, thickness=4)
        
        # Display text about confidence rate above each box
        text = f'{confidences[i]:.2f}'
        ### YOUR CODE HERE
        # cv2.putText(result,text,(left,top),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA)
        crop = crop[(top-10):(top+height+10),(left-10):(left+width+10)]
        crop=load_and_preprocess(crop)
        knn_from_joblib.predict([crop])
        text3 =f'{label_dict[knn_from_joblib.predict([crop]).argmax()]}'
        cv2.putText(result,text3,(box[0],box[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA)
    # Display text about number of detected faces on topleft corner
    # YOUR CODE HERE
    text2= f'Number of face detected {len(final_boxes)}'
    cv2.putText(result,text2,(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA)

    # cv2.imshow('face detection', result)
    return result,final_boxes
# Open Webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()

    # frame is now the image capture by the webcam (one frame of the video)
    outs= get_predict_yolo(frame)
    boxes,confidences=threshold_boxes(outs)
    result,final_boxes=nms_boxes(boxes,confidences)
    cv2.imshow('face detection', result)
    c = cv2.waitKey(1)
		# Break when pressing ESC
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()