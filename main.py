import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

# CONSTANTS
VID_WIDTH = 800 # the width of the video feed displayed
VID_HEIGHT = 600 # the height of the video feed displayed
PROX_THRES = 35 # how close the robot needs to be to an object to count it as disinfected [in pixels]
                # aka the range of the robot
FUZZY = 20 # how much tolerance to bounding box detections to suppress it [in pixels]

# SET to keep track of which objects were already disinfected
vis = set() # tuples of (xmin, ymin, xmax, ymax)

# Robot position and velocity
robot_x = 235
robot_y = 480
robot_x_vel = 0
robot_y_vel = -6
robot_sz = 25

# Coordinates from basement
quad_coords = {
    "lonlat": np.array([
        [0, 0], # Bottom right
        [246, 5], # Bottom left
        [154, 179], # Top left
        [25, 173] # Top right
    ]),
    "pixel": np.array([
        [797, 598], # Bottom right
        [75, 480], # Bottom left
        [450, 317], # Top left
        [721, 343] # Top right
    ])
}

## Pixel mapper from camera feed to longitutde latitude
class PixelMapper(object):
    """
    Create an object for converting pixels to geographic coordinates,
    using four points with known locations which form a quadrilteral in both planes
    Parameters
    ----------
    pixel_array : (4,2) shape numpy array
        The (x,y) pixel coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    lonlat_array : (4,2) shape numpy array
        The (lon, lat) coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    """
    def __init__(self, pixel_array, lonlat_array):
        assert pixel_array.shape==(4,2), "Need (4,2) input array"
        assert lonlat_array.shape==(4,2), "Need (4,2) input array"
        self.M = cv2.getPerspectiveTransform(np.float32(pixel_array),np.float32(lonlat_array))
        self.invM = cv2.getPerspectiveTransform(np.float32(lonlat_array),np.float32(pixel_array))
        
    def pixel_to_lonlat(self, pixel):
        """
        Convert a set of pixel coordinates to lon-lat coordinates
        Parameters
        ----------
        pixel : (N,2) numpy array or (x,y) tuple
            The (x,y) pixel coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (lon, lat) coordinates
        """
        if type(pixel) != np.ndarray:
            pixel = np.array(pixel).reshape(1,2)
        assert pixel.shape[1]==2, "Need (N,2) input array" 
        pixel = np.concatenate([pixel, np.ones((pixel.shape[0],1))], axis=1)
        lonlat = np.dot(self.M,pixel.T)
        
        return (lonlat[:2,:]/lonlat[2,:]).T
    
    def lonlat_to_pixel(self, lonlat):
        """
        Convert a set of lon-lat coordinates to pixel coordinates
        Parameters
        ----------
        lonlat : (N,2) numpy array or (x,y) tuple
            The (lon,lat) coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (x, y) pixel coordinates
        """
        if type(lonlat) != np.ndarray:
            lonlat = np.array(lonlat).reshape(1,2)
        assert lonlat.shape[1]==2, "Need (N,2) input array" 
        lonlat = np.concatenate([lonlat, np.ones((lonlat.shape[0],1))], axis=1)
        pixel = np.dot(self.invM,lonlat.T)
        
        return (pixel[:2,:]/pixel[2,:]).T

## Distance measurement method
def dist(x1, x2):
    """Takes 2 pairs of cooridninates, and finds Euclidean distance between each pair and avg"""
    ed1 = (x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2 
    ed1 = math.sqrt(ed1)

    ed2 = (x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2 
    ed2 = math.sqrt(ed2)

    return (ed1 + ed2) / 2

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file("my_ssd_mobnet/pipeline.config")
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join("my_ssd_mobnet", 'ckpt-5')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap("my_ssd_mobnet/label_map.pbtxt")

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get scaling ratios
SW = width / VID_WIDTH
SH = height / VID_HEIGHT

while cap.isOpened(): 
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    image_np_with_detections_copy = image_np.copy()

    # Get the bounding box coodinates and names
    coords = viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections_copy,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)

    # Draw in alignment points for 2D perspective transform from feed to floor
    #cv2.circle(image_np_with_detections, (int(797 * SW), int(598 * SH)), radius=2, color=(0, 0, 255), thickness=-1)
    #cv2.circle(image_np_with_detections, (int(75 * SW), int(480 * SH)), radius=2, color=(0, 0, 255), thickness=-1)
    #cv2.circle(image_np_with_detections, (int(450 * SW), int(317 * SH)), radius=2, color=(0, 0, 255), thickness=-1)
    #cv2.circle(image_np_with_detections, (int(721 * SW), int(343 * SH)), radius=2, color=(0, 0, 255), thickness=-1)
    
    # Initialize pixel mapper
    pm = PixelMapper(quad_coords["pixel"], quad_coords["lonlat"])

    # Display the real world location on the floor for each detected item
    for item in coords:
        # Item: [name, xmin, ymin, xmax, ymax]
        name, xmin, ymin, xmax, ymax = item[0], item[1], item[2], item[3], item[4]
        pos = (xmin, ymin, xmax, ymax)
        #print("DEBUG:", name, xmin, ymin, xmax, ymax)

        # Find the centroid of the bounding box
        centroid_x = (xmin + xmax) / 2
        centroid_y = (ymin + ymax) / 2

        # Draw up the robot
        cv2.rectangle(image_np_with_detections, (robot_x - robot_sz // 2, robot_y - robot_sz // 2), (robot_x + robot_sz // 2, robot_y - robot_sz // 2), (255, 0, 0), 2)
        cv2.circle(image_np_with_detections, (robot_x, robot_y), radius = robot_sz // 2, color = (255, 0, 0), thickness = 2)
        cv2.putText(image_np_with_detections, "Robot", (robot_x - robot_sz // 2, robot_y - robot_sz // 2 - 25),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Check robot proximity to each detected object
        d = math.sqrt((robot_x - centroid_x) ** 2 + (robot_y - centroid_y) ** 2)
        print("DEBUG:", d, robot_x, centroid_x, robot_y, centroid_y)
        
        if d <= PROX_THRES:
            # robot has cleaned this object
            vis.add(pos)

        # Check if this item has already been disinfected
        cleaned = False
        for x in vis:
            if dist(x, pos) <= float(FUZZY):
                # Already cleaned
                cleaned = True
                break
        
        # Draw a green bounding box if object is disinfected
        # A red if the object is not
        green = (0, 255, 0)
        red = (0, 0, 255)
        if cleaned:
            lbl = name + " DISINFECTED"
            cv2.rectangle(image_np_with_detections, (xmin, ymin), (xmax, ymax), green, 2)
            cv2.putText(image_np_with_detections, lbl, (xmin, ymin - 25),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
        else:
            lbl = name + " DIRTY"
            cv2.rectangle(image_np_with_detections, (xmin, ymin), (xmax, ymax), red, 2)
            cv2.putText(image_np_with_detections, lbl, (xmin, ymin - 25),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)

        # Map this to the real world
        tmp = pm.pixel_to_lonlat((centroid_x * 1 / SW, centroid_y * 1 / SH))
        vid_x = int(centroid_x)
        vid_y = int(centroid_y)
        real_x = int(tmp[0][0] * SW)
        real_y = int(tmp[0][1] * SH)

        # Display the transformed centroid on the bounding box
        # And draw the centroid
        label = name + " IRL Pos (cm): " + str(real_x) + " " + str(real_y)
        cv2.circle(image_np_with_detections, (vid_x, vid_y), radius = 3, color = (0, 0, 0), thickness = 1)
        cv2.putText(image_np_with_detections, label, (vid_x, vid_y+50),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Update robot position
    if robot_y < 300:
        robot_x_vel = 5
        robot_y_vel = 0
    robot_x += robot_x_vel
    robot_y += robot_y_vel

    # Display the resulting frame
    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (VID_WIDTH, VID_HEIGHT)))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break