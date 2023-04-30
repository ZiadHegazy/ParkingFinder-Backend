#!pip install -U torch==1.13.1 torchvision==0.14.1 -f https://download.pytorch.org/whl/cu101/torch_stable.html
#!pip install cython pyyaml==5.3.1
#!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
import cv2
import torch, torchvision
import re
print(torch.__version__, torch.cuda.is_available())
#!gcc --version
# opencv is pre-installed on colab
#!python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
#import cv2
import random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
#from detectron2.evaluation.coco_evaluation import COCOEvaluator
import os


#!curl -L "https://universe.roboflow.com/ds/ZpXziOCQPR?key=vZ3oYeBWvT" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)

cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001


cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1500 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05




cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4 #your number of classes + 1

cfg.TEST.EVAL_PERIOD = 500

cfg.MODEL.WEIGHTS = os.path.join("D:/bachelor/datasets/mask r cnn coco", "model_final.pth")
cfg.DATASETS.TEST = ("my_dataset_test", )
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.MODEL.DEVICE='cpu'
predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get("my_dataset_test")

from detectron2.utils.visualizer import ColorMode
import glob

from detectron2.utils.visualizer import ColorMode
import glob
import numpy as np
from scipy.optimize import minimize_scalar


def distance_to_cubic_bezier(point, curve):
    # define the objective function to minimize
    def objective(t):
        return np.linalg.norm(curve_point(t, curve) - point)

    # find the parameter value t that minimizes the objective function
    res = minimize_scalar(objective, bounds=(0, 1), method='bounded')
    t = res.x

    # calculate the closest point on the curve to the given point
    closest_point = curve_point(t, curve)

    # calculate the distance between the closest point and the given point
    distance = np.linalg.norm(closest_point - point)

    return distance


def curve_point(t, curve):
    # evaluate the cubic Bezier curve at parameter value t
    p0, p1, p2, p3 = curve
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3






import numpy as np

def solve_cubic(a, b, c, d):
    """
    Solve the equation ax^3 + bx^2 + cx + d = 0 using Cardano's formula.
    """
    q = (3 * a * c - b ** 2) / (9 * a ** 2)
    r = (9 * a * b * c - 27 * a ** 2 * d - 2 * b ** 3) / (54 * a ** 3)
    delta = q ** 3 + r ** 2
    if delta > 0:
        s = np.cbrt(r + np.sqrt(delta))
        t = np.cbrt(r - np.sqrt(delta))
        return -b / (3 * a) + s + t
    elif delta == 0:
        if r == 0:
            return -b / (3 * a)
        else:
            return -b / (3 * a) + 2 * r / (3 * a * np.cbrt(q))
    else:
        theta = np.arccos(r / np.sqrt(-(q ** 3)))
        return -b / (3 * a) + 2 * np.sqrt(-q) * np.cos(theta / 3)

def get_t_for_x(x, p0, p1, p2, p3):
    """
    Get the parameter t for a given x coordinate on a cubic Bezier curve.
    """
    a = -p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]
    b = 3 * p0[0] - 6 * p1[0] + 3 * p2[0]
    c = -3 * p0[0] + 3 * p1[0]
    d = p0[0] - x
    t = solve_cubic(a, b, c, d)
    return t


import numpy as np


def find_t_for_x(x, P):
    """
    Find the parameter t for a given x coordinate on a cubic Bezier curve.
    """
    # Extract the control points from the input array P.
    P0, P1, P2, P3 = P

    # Define the coefficients of the cubic Bezier curve in x.
    cx = -P0[0] + 3 * P1[0] - 3 * P2[0] + P3[0]
    bx = 3 * P0[0] - 6 * P1[0] + 3 * P2[0]
    ax = -3 * P0[0] + 3 * P1[0]

    # Define the roots of the cubic equation in x.
    roots = np.roots([ax, bx, cx - x])

    # Find the real root in the range [0,1].
    for r in roots:
        if 0 <= r <= 1 and np.isreal(r):
            t = r.real
            # Calculate the corresponding y coordinate using the Bezier curve equation.
            y = (1 - t) ** 3 * P0[1] + 3 * (1 - t) ** 2 * t * P1[1] + 3 * (1 - t) * t ** 2 * P2[1] + t ** 3 * P3[1]
            return t, y

    # If no real root is found, return None.
    return None
def getNumbers(img,control,capacity,location,name,lat,lng):
    imageName = "D:/bachelor/datasets/{}".format(img)
    im = cv2.imread(imageName)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=test_metadata,
                   scale=0.8
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    print(outputs["instances"].scores)
    print(float(outputs["instances"].pred_boxes.get_centers()[0][0]))
    parked = 0
    for center in outputs["instances"].pred_boxes.get_centers():
        point = np.array([float(center[0]), -float(center[1])])
        controlPoints = (
            np.array(control[0]), np.array(control[1]), np.array(control[2]),
            np.array(control[3]))
        distance = distance_to_cubic_bezier(point, controlPoints)
        print(distance)
        if (distance <= 39):
            parked += 1;
    print("parked car" + str(parked))
    available=capacity-parked
    return {"available":available,"occupied":parked,"location":location,"name":name,"lat":lat,"lng":lng}

from flask import Flask, request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from flask_jsonpify import jsonify
from pymongo import MongoClient
client = MongoClient('mongodb+srv://ziad:ZAheg1234@cluster0.ceczioq.mongodb.net/test')
db = client["test"]
locations=db["locations"]
app = Flask(__name__)
api = Api(app)
# list=[{"img":"img14.jpg","locationName":"Right Side","capacity":5,"googleMaps":"Link1","controlPoints":[
#             [2.24, -190.19], [190.72, -145.6], [316.544, -47.513],
#             [633.6, -68.273]]} ,{"img":"img12.jpg","locationName":"Left Side","capacity":5,"googleMaps":"Link2","controlPoints":[
#             [2.944,-163.28], [81.536,-89.24], [433.92,-73.526],
#             [640,-105.41]]}]
# location1={"location":"Palm resort 1st settlement street 36 villa 556" , "parkings":list}
#
# locations.insert_one(location1)

@app.route("/<name>")
def home(name):
    parkings=[]
    final=[]
    for x in locations.find():

        if(re.sub(",","",(name.lower())) in x.get('location').lower()):
            parkings=x.get("parkings")
    for x in parkings:
        result=getNumbers(x.get("img"),x.get("controlPoints"),x.get("capacity"),x.get("googleMaps"),x.get("locationName"),x.get("lat"),
                          x.get("lng"))
        final.append(result)
    return  jsonify(result=final)
if __name__ == '__main__':
     app.run(debug=True ,port=8080,use_reloader=False)