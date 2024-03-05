<!-- ORIGINAL -->

import tensorflow as tf
import cv2
import numpy as np
import tensorflow_hub as hub
import random
import math

# Load the model directly from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1").signatures["default"]

colorcodes = {}
def drawbox(image, ymin, xmin, ymax, xmax, namewithscore, color):
    
    im_height, im_width, _  = image.shape
    left,top,right,bottom = int(xmin*im_width), int(ymin*im_height), int(xmax*im_width),int(ymax*im_height)
    cv2.rectangle(image,(left,top),(right,bottom),color = color,thickness = 1 )
    FONT_SCALE = 5e-3
    THICKNESS_SCALE = 4e-3
    width = right-left
    height = bottom-top
    TEXT_Y_OFFSET_SCALE = 1e-2
    cv2.rectangle(
        image,
        (left,top- int(height * 6e-2)),
        (right,top),
        color = color,
        thickness = -1
        
    )
    cv2.putText(
        image,
        namewithscore,
        (left,top-int(height * TEXT_Y_OFFSET_SCALE)),
        fontFace = cv2.FONT_HERSHEY_PLAIN,
        fontScale = min(width,height)* FONT_SCALE,
        thickness = math.ceil(min(width,height)* THICKNESS_SCALE),
        color = (255,255,255)
    )
def draw(image, boxes, classnames, scores):
    boxesidx = tf.image.non_max_suppression(boxes, scores, max_output_size=20, score_threshold=0.2)
    for i in boxesidx:
        ymin, xmin, ymax, xmax = tuple(boxes[i])
        classname = classnames[i].decode("ascii")
        if classname in colorcodes.keys():
            color = colorcodes[classname]
        else:
            c1 = random.randrange(0, 255, 30)
            c2 = random.randrange(0, 255, 25)
            c3 = random.randrange(0, 255, 50)
            colorcodes.update({classname: (c1, c2, c3)})
            color = colorcodes[classname]
        namewithscore = "{}:{}".format(classname, int(100 * scores[i]))
        drawbox(image, ymin, xmin, ymax, xmax, namewithscore, color)

    return image

video = cv2.VideoCapture('image3.jpg')
while True:
    _, img = video.read()
    img = cv2.resize(img, (900, 700))
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)[tf.newaxis, ...]
    detection = model(img2)
    result = {key: value.numpy() for key, value in detection.items()}
    imagewithboxes = draw(img, result['detection_boxes'], result['detection_class_entities'], result["detection_scores"])
    cv2.imshow("detection", imagewithboxes)
    cv2.waitKey(27) & 0xFF == ord('q'):
        break
    
















<!-- Hitung DOOR -->
import tensorflow as tf
import cv2
import numpy as np
import tensorflow_hub as hub
import random
import math

# Load the model directly from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1").signatures["default"]

colorcodes = {}

# Initialize door state variables
door_open = False
door_open_counter = 0

def drawbox(image, ymin, xmin, ymax, xmax, namewithscore, color):
    # Function to draw bounding box on the image
    im_height, im_width, _  = image.shape
    left, top, right, bottom = int(xmin * im_width), int(ymin * im_height), int(xmax * im_width), int(ymax * im_height)
    cv2.rectangle(image, (left, top), (right, bottom), color=color, thickness=1)
    FONT_SCALE = 5e-3
    THICKNESS_SCALE = 4e-3
    width = right - left
    height = bottom - top
    TEXT_Y_OFFSET_SCALE = 1e-2
    cv2.rectangle(
        image,
        (left, top - int(height * 6e-2)),
        (right, top),
        color=color,
        thickness=-1
    )
    cv2.putText(
        image,
        namewithscore,
        (left, top - int(height * TEXT_Y_OFFSET_SCALE)),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=min(width, height) * FONT_SCALE,
        thickness=math.ceil(min(width, height) * THICKNESS_SCALE),
        color=(255, 255, 255)
    )

def draw(image, boxes, classnames, scores):
    global door_open, door_open_counter
    
    boxesidx = tf.image.non_max_suppression(boxes, scores, max_output_size=20, score_threshold=0.2)
    for i in boxesidx:
        ymin, xmin, ymax, xmax = tuple(boxes[i])
        classname = classnames[i].decode("ascii")
        if classname in colorcodes.keys():
            color = colorcodes[classname]
        else:
            c1 = random.randrange(0, 255, 30)
            c2 = random.randrange(0, 255, 25)
            c3 = random.randrange(0, 255, 50)
            colorcodes.update({classname: (c1, c2, c3)})
            color = colorcodes[classname]
        
        namewithscore = "{}:{}".format(classname, int(100 * scores[i]))
        drawbox(image, ymin, xmin, ymax, xmax, namewithscore, color)
        
        if classname.lower() == 'person' or classname.lower() == 'door':
            # Assuming opening the door is detected when a person or door is detected
            if not door_open:
                door_open = True
                door_open_counter += 1
                print(f"Door opened! Total opens: {door_open_counter}")
    return image

video = cv2.VideoCapture(0)
while True:
    _, img = video.read()
    img = cv2.resize(img, (900, 700))
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)[tf.newaxis, ...]
    detection = model(img2)
    result = {key: value.numpy() for key, value in detection.items()}
    imagewithboxes = draw(img, result['detection_boxes'], result['detection_class_entities'], result["detection_scores"])
    cv2.imshow("detection", imagewithboxes)
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break












<!-- hitung door V2 -->
import tensorflow as tf
import cv2
import numpy as np
import tensorflow_hub as hub
import random
import math

# Load the model directly from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1").signatures["default"]

colorcodes = {}

# Initialize door state variables
door_open = False
door_open_counter = 0

def drawbox(image, ymin, xmin, ymax, xmax, namewithscore, color):
    # Function to draw bounding box on the image
    im_height, im_width, _  = image.shape
    left, top, right, bottom = int(xmin * im_width), int(ymin * im_height), int(xmax * im_width), int(ymax * im_height)
    cv2.rectangle(image, (left, top), (right, bottom), color=color, thickness=1)
    FONT_SCALE = 5e-3
    THICKNESS_SCALE = 4e-3
    width = right - left
    height = bottom - top
    TEXT_Y_OFFSET_SCALE = 1e-2
    cv2.rectangle(
        image,
        (left, top - int(height * 6e-2)),
        (right, top),
        color=color,
        thickness=-1
    )
    cv2.putText(
        image,
        namewithscore,
        (left, top - int(height * TEXT_Y_OFFSET_SCALE)),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=min(width, height) * FONT_SCALE,
        thickness=math.ceil(min(width, height) * THICKNESS_SCALE),
        color=(255, 255, 255)
    )

def draw(image, boxes, classnames, scores):
    global door_open, door_open_counter
    
    boxesidx = tf.image.non_max_suppression(boxes, scores, max_output_size=20, score_threshold=0.2)
    for i in boxesidx:
        ymin, xmin, ymax, xmax = tuple(boxes[i])
        classname = classnames[i].decode("ascii")
        if classname in colorcodes.keys():
            color = colorcodes[classname]
        else:
            c1 = random.randrange(0, 255, 30)
            c2 = random.randrange(0, 255, 25)
            c3 = random.randrange(0, 255, 50)
            colorcodes.update({classname: (c1, c2, c3)})
            color = colorcodes[classname]
        
        namewithscore = "{}:{}".format(classname, int(100 * scores[i]))
        drawbox(image, ymin, xmin, ymax, xmax, namewithscore, color)
        
        if classname.lower() == 'personiis' or classname.lower() == 'door':
            # Assuming opening the door is detected when a person or door is detected
            if not door_open:
                door_open = True
                door_open_counter += 1
                print(f"Door opened! Total opens: {door_open_counter}")
    return image

video = cv2.VideoCapture(0)
while True:
    _, img = video.read()
    img = cv2.resize(img, (900, 700))
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)[tf.newaxis, ...]
    detection = model(img2)
    result = {key: value.numpy() for key, value in detection.items()}
    imagewithboxes = draw(img, result['detection_boxes'], result['detection_class_entities'], result["detection_scores"])
    cv2.imshow("detection", imagewithboxes)
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

















<!-- Door open v3 -->
import tensorflow as tf
import cv2
import numpy as np
import tensorflow_hub as hub
import random
import math

# Load the model directly from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1").signatures["default"]

colorcodes = {}

# Initialize door state variables
door_open = False
door_open_counter = 0

def drawbox(image, ymin, xmin, ymax, xmax, namewithscore, color):
    # Function to draw bounding box on the image
    im_height, im_width, _  = image.shape
    left, top, right, bottom = int(xmin * im_width), int(ymin * im_height), int(xmax * im_width), int(ymax * im_height)
    cv2.rectangle(image, (left, top), (right, bottom), color=color, thickness=1)
    FONT_SCALE = 5e-3
    THICKNESS_SCALE = 4e-3
    width = right - left
    height = bottom - top
    TEXT_Y_OFFSET_SCALE = 1e-2
    cv2.rectangle(
        image,
        (left, top - int(height * 6e-2)),
        (right, top),
        color=color,
        thickness=-1
    )
    cv2.putText(
        image,
        namewithscore,
        (left, top - int(height * TEXT_Y_OFFSET_SCALE)),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=min(width, height) * FONT_SCALE,
        thickness=math.ceil(min(width, height) * THICKNESS_SCALE),
        color=(255, 255, 255)
    )

def draw(image, boxes, classnames, scores):
    global door_open, door_open_counter
    
    boxesidx = tf.image.non_max_suppression(boxes, scores, max_output_size=20, score_threshold=0.2)
    for i in boxesidx:
        ymin, xmin, ymax, xmax = tuple(boxes[i])
        classname = classnames[i].decode("ascii")
        print(f"Detected class: {classname}")  # Check detected class
        if classname in colorcodes.keys():
            color = colorcodes[classname]
        else:
            c1 = random.randrange(0, 255, 30)
            c2 = random.randrange(0, 255, 25)
            c3 = random.randrange(0, 255, 50)
            colorcodes.update({classname: (c1, c2, c3)})
            color = colorcodes[classname]
        
        namewithscore = "{}:{}".format(classname, int(100 * scores[i]))
        drawbox(image, ymin, xmin, ymax, xmax, namewithscore, color)
        
        if classname.lower() == 'person' or classname.lower() == 'door':
            # Assuming opening the door is detected when a person or door is detected
            if not door_open:
                door_open = True
                door_open_counter += 1
                print(f"Door opened! Total opens: {door_open_counter}")
                # Reset door_open to False to allow counting for the next door
                door_open = False

    # Print information for each frame
    print(f"Current state: Door open - {door_open}, Total opens: {door_open_counter}")

    return image

video = cv2.VideoCapture('vidio/vid3.mp4')
while True:
    ret, img = video.read()
    
    # Check if frame is not empty
    if not ret or img is None:
        break
    
    img = cv2.resize(img, (600, 500))
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)[tf.newaxis, ...]
    detection = model(img2)
    result = {key: value.numpy() for key, value in detection.items()}
    imagewithboxes = draw(img, result['detection_boxes'], result['detection_class_entities'], result["detection_scores"])
    cv2.imshow("detection", imagewithboxes)
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break
