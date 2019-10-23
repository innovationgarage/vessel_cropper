import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def showImage(img):
    plt.figure()
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def cropImage(img, bbox, margin, output_size):
    h, w, ch = img.shape
    left, right, top, bottom = bbox
    if left - margin < 0:
        left = 0
    else:
        left = left - margin
    if right + margin > w:
        rght = w
    else:
        right = right + margin
    if top - margin < 0:
        top = 0
    else:
        top = top - margin
    if bottom + margin > h:
        bottom = h
    else:
        bottom = bottom + margin
    cropped = img[top:bottom, left:right]
    cropped = cv2.resize(cropped,output_size, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('cropped', cropped)
    cv2.waitKey()

def detectBoat(img):
    t0 = time.time()
    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()

    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.8:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            category = int(detection[1])
            if category == 8: #boat
                return (left, right, top, bottom)
            else:
                return None

def calculateAR(record):
    width  = record['bbox'][1]-record['bbox'][0]
    height = record['bbox'][3]-record['bbox'][2]
    aspect_ratio = width/height
    record['ar'] = aspect_ratio
    return record

def chooseImage(records, goal_ar=1):
    ims = [im for im in records]
    ars = np.array([np.abs(1-records[im]['ar']) for im in records])
    closest_ar = np.argmin(ars)
    return ims[closest_ar]

if __name__=="__main__":
    OUTPUT_SIZE = (152, 152)
    INPUT_DIR = "saghar/"
    OUTPUT_DIR = "out/"
    ASPECT_RATIO = 1
    MARGIN = 50
    
    frozen_weights = "/home/saghar/IG/projects/epimp-brain/TFModels/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
    model_config = "/home/saghar/IG/projects/epimp-brain/TFModels/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt"
    cvNet = cv2.dnn.readNetFromTensorflow(frozen_weights, model_config)

    images = os.listdir(INPUT_DIR)
    records = {}
    for image in images:
        img = cv2.imread(os.path.join(INPUT_DIR, image))
        detection = detectBoat(img)
        if detection:
            records[image] = {'bbox': detectBoat(img)}
            records[image] = calculateAR(records[image])

    image_to_use = chooseImage(records, ASPECT_RATIO)
    img = cv2.imread(os.path.join(INPUT_DIR, image_to_use))
    bbox = [int(el) for el in records[image_to_use]['bbox']]
    cropImage(img, bbox, MARGIN, OUTPUT_SIZE)    
