# Load YOLOv8n, train it on COCO128 for 3 epochs and predict an image with it
from ultralytics import YOLO
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


model = YOLO("/content/last.pt")  # load a pretrained YOLOv8n detection model
# model.train(data='coco128.yaml', epochs=3)  # train the model
# model('https://ultralytics.com/images/bus.jpg')  # predict on an image
model("/content/img12.jpg", save_txt=True, save=True, save_conf=True)

with open("/content/runs/detect/predict/labels/img12.txt", "r") as file:
    # Loop through each line in the file
    for line in file:
        arr = line.split(' ')
        xcenter = float(arr[1])
        ycenter = float(arr[2])
        point = np.array([xcenter, -ycenter])
        controlPoints = (
        np.array([0.0046, -0.842]), np.array([0.1274, -0.46]), np.array([0.678, -0.379]), np.array([1, -0.543]))
        distance = distance_to_cubic_bezier(point, controlPoints)
        print(distance)




img14
# Load YOLOv8n, train it on COCO128 for 3 epochs and predict an image with it
from ultralytics import YOLO
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
    return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3

model = YOLO("/content/last.pt")  # load a pretrained YOLOv8n detection model
#model.train(data='coco128.yaml', epochs=3)  # train the model
#model('https://ultralytics.com/images/bus.jpg')  # predict on an image
model("/content/img4.jpg",save_txt=True,save=True,save_conf=True)
parked=0
with open("/content/runs/detect/predict8/labels/img4.txt", "r") as file:
    # Loop through each line in the file
    for line in file:
        arr=line.split(' ')
        xcenter = float(arr[1])
        ycenter = float(arr[2])
        point=np.array([xcenter,-ycenter])
        controlPoints=(np.array([0.0035,-0.91]),np.array([0.298,-0.697]),np.array([0.4946,-0.227]),np.array([0.99,-0.327]))
        distance=distance_to_cubic_bezier(point,controlPoints)
        if(distance<=0.19):
          parked+=1;
        print(arr[5])
    print("parked car"+parked)

#########3
from PIL import Image

# Load the image
image = Image.open("image.jpg")

# Get the dimensions of the image
width, height = image.size

# Calculate the size of each quarter
quarter_width = width // 2
quarter_height = height // 2

# Divide the image into four quarters
top_left = image.crop((0, 0, quarter_width, quarter_height))
top_right = image.crop((quarter_width, 0, width, quarter_height))
bottom_left = image.crop((0, quarter_height, quarter_width, height))
bottom_right = image.crop((quarter_width, quarter_height, width, height))

# Save the quarters as separate images
top_left.save("top_left.jpg")
top_right.save("top_right.jpg")
bottom_left.save("bottom_left.jpg")
bottom_right.save("bottom_right.jpg")
model("/content/top_left.jpg",save_txt=True,save=True,save_conf=True)
model("/content/top_right.jpg",save_txt=True,save=True,save_conf=True)
model("/content/bottom_left.jpg",save_txt=True,save=True,save_conf=True)
model("/content/bottom_right.jpg",save_txt=True,save=True,save_conf=True)


with open("/content/runs/detect/predict/labels/img12.txt", "r") as file:
    # Loop through each line in the file
    countFire=0
    countSmoke=0
    for line in file:
        arr = line.split(' ')
        xcenter = float(arr[1])
        ycenter = float(arr[2])
        if(arr[0]=="0"):
            countFire+=1
        else
            countSmoke+=1


#!curl -L "https://universe.roboflow.com/ds/mp0lgS4Z8q?key=tSM4XCYvA4" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
logger = 'Comet' #@param ['Comet', 'ClearML', 'TensorBoard']

if logger == 'Comet':
  %pip install -q comet_ml
  import comet_ml; comet_ml.init()
elif logger == 'ClearML':
  %pip install -q clearml
  import clearml; clearml.browser_login()
elif logger == 'TensorBoard':
  %load_ext tensorboard
  %tensorboard --logdir runs/train