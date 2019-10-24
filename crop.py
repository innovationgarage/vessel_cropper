import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
from collect_data import *
import requests
import shutil
import tempfile

def cropImage(img, bbox, margin, output_size, imo_no, output_dir):
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
    if top - margin*2 < 0:
        top = 0
    else:
        top = int(top - margin*2)
    if bottom + margin/2 > h:
        bottom = h
    else:
        bottom = int(bottom + margin/2)
    cropped = img[top:bottom, left:right]
#    cropped = cv2.resize(cropped,output_size, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(output_dir, '{}.jpg'.format(imo_no)), cropped)

def detectBoat(img, cvNet):
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

def downloadImage(image_url, local_path):
    resp = requests.get(image_url, stream=True)
    if resp.ok:
        local_file = open(local_path, "wb")
        resp.raw.decode_content = True
        shutil.copyfileobj(resp.raw, local_file)
        del resp
        return True
    else:
        return False

def crop(IMO):
    OUTPUT_SIZE = (152, 152)
    INPUT_DIR = 'temp' #temporary
    OUTPUT_DIR = "out/"
    ASPECT_RATIO = 1
    MARGIN = 50

    start = time.time()
    #Config the model
    frozen_weights = "model/frozen_inference_graph.pb"
    model_config = "model/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt"
    cvNet = cv2.dnn.readNetFromTensorflow(frozen_weights, model_config)

    #Get the ship gallery from ES & download photos
    image_urls = findShip(IMO)
    if image_urls:
        with tempfile.TemporaryDirectory(dir=INPUT_DIR) as tmpdirname:
            # print('created temporary directory', tmpdirname)
            for i, image_url in enumerate(image_urls[:10]):
                res = downloadImage(image_url, os.path.join(tmpdirname, '{}.jpg'.format(i)))
                # print(image_url, res)
            #Run ship photos through an object detector and get bboxes
            images = os.listdir(tmpdirname)
            records = {}
            for image in images:
                img = cv2.imread(os.path.join(tmpdirname, image))
                detection = detectBoat(img, cvNet)
                if detection:
                    records[image] = {'bbox': detectBoat(img, cvNet)}
                    records[image] = calculateAR(records[image])

            #Choose the most relevant photo and crop it to the best region and size
            image_to_use = chooseImage(records, ASPECT_RATIO)
            img = cv2.imread(os.path.join(tmpdirname, image_to_use))
            bbox = [int(el) for el in records[image_to_use]['bbox']]
            cropImage(img, bbox, MARGIN, OUTPUT_SIZE, IMO, OUTPUT_DIR)
        end = time.time()
        time_elapsed = end - start
        print("Total elapsed time was {} seconds.".format(time_elapsed))
        return os.path.join(OUTPUT_DIR, '{}.jpg'.format(IMO))
    else:
        print('I found nothing!')
        return None

def test(test_path):
    OUTPUT_SIZE = (152, 152)
    INPUT_DIR = 'temp' #temporary
    OUTPUT_DIR = "out/"
    ASPECT_RATIO = 1
    MARGIN = 50
    IMO = 'tst'
    
    start = time.time()
    #Config the model
    frozen_weights = "model/frozen_inference_graph.pb"
    model_config = "model/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt"
    cvNet = cv2.dnn.readNetFromTensorflow(frozen_weights, model_config)

    #Get the ship gallery from ES & download photos
    images = os.listdir(test_path)
    tmpdirname = test_path

    if images:
        records = {}
        for image in images:
            img = cv2.imread(os.path.join(tmpdirname, image))
            detection = detectBoat(img, cvNet)
            if detection:
                records[image] = {'bbox': detectBoat(img, cvNet)}
                records[image] = calculateAR(records[image])

        print(images, records)
        #Choose the most relevant photo and crop it to the best region and size
        image_to_use = chooseImage(records, ASPECT_RATIO)
        img = cv2.imread(os.path.join(tmpdirname, image_to_use))
        bbox = [int(el) for el in records[image_to_use]['bbox']]
        cropImage(img, bbox, MARGIN, OUTPUT_SIZE, IMO, OUTPUT_DIR)
        end = time.time()
        time_elapsed = end - start
        print("Total elapsed time was {} seconds.".format(time_elapsed))
        return os.path.join(OUTPUT_DIR, '{}.jpg'.format(IMO))
    else:
        print('I found nothing!')
        return None
    
if __name__=="__main__":
    test_path = sys.argv[1]
    outfile = test(test_path)
