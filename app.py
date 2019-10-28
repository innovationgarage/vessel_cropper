from flask import Flask
app = Flask(__name__)
from flask import request
from flask import send_file
import crop

@app.route('/')
def home():
    IMO = request.args.get("imo")
    thumbnail = request.args.get("thumbnail")

    outfile = crop.crop(IMO, thumbnail, cvNet)
    if outfile:
        return send_file(outfile, mimetype="image/jpg")
    else:
        return None

import cv2
#Config the model
frozen_weights = "model/frozen_inference_graph.pb"
model_config = "model/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt"
cvNet = cv2.dnn.readNetFromTensorflow(frozen_weights, model_config)

if __name__=="__main__":
    #Config the model
    app.run(host='0.0.0.0', debug=True)

    
