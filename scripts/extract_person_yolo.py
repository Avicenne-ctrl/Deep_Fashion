from ultralytics import YOLO
import numpy
import cv2
import numpy as np
from typing import List, Dict
from roboflow import Roboflow
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os
import pandas as pd

def bounding_yolovn(bounding_box: List[float], image: numpy.ndarray):
    """
        Given a bounding boxe and the corresponding image
        return the image center on the bounding boxe
        
        Args:
            bounding_box (List[float]):
                the bounding boxe
                
            image (numpy.ndarray):
                the image we want 
                
        Returns:
            the image centered 
            
        Raise:
        -----
            - if no objects in bounding box
            - if bounding box is not a list
    
    """
    if len(bounding_box) == 0:
        raise TypeError(f"no bounding box found: {bounding_box}")
    
    if not isinstance(bounding_box, list):
        raise TypeError(f"wrong type object given, expected list found : {type(bounding_box).__name__}")
    
    if not isinstance(image, numpy.ndarray):
        raise TypeError(f"wrong type object given, expected numpy.ndarray found : {type(image).__name__}")
    
    x_min, y_min, x_max, y_max =  bounding_box
    centered_image = image[int(y_min) : int(y_max), int(x_min) : int(x_max)]
    return centered_image


def extract_boxes_yolovn(model: YOLO, image: numpy.ndarray):#->List[List[float]]:
    """
        Detect bounding boxes using yolo model

        Args:
            model (YOLO): 
                the yolo model we will use
                
            image (numpy.ndarray): 
                the image we want to extract objects

        Returns:
            bounding_boxes (List[List[float]]): 
                list of tuple for bounding boxes
            
            label_boxes (List[str]): 
                list of label for each detected objects
                
        Raise:
        ------
            - if model is not an ultralytics.YOLO object
            
        Example:
            >>> extract_boxes_yolovn(YOLO_MODEL, image = cv2.imread(your_img))
            >>> bounding_boxes = [[x_min, y_min, x_max, y_max], ...]
            >>> label_boxes = ["label1", ...]
    """
    
    if not isinstance(model, YOLO):
        raise TypeError(f"wrong type object given, expected YOLO found : {type(model).__name__}")
    
    bounding_boxes = []
    label_boxes    = []
    results = model.predict(image, verbose=False)
    for res in results:
        bounding_boxes.append(res.boxes.xyxy)
        label_boxes.append(res.boxes.cls.tolist())  # .cls contains the labels of the boxes
                
    return bounding_boxes[0].tolist(), label_boxes[0]

def bounding_yolovn(bounding_box: List[float], image: numpy.ndarray):
    """
        Given a bounding boxe and the corresponding image
        return the image center on the bounding boxe
        
        Args:
            bounding_box (List[float]):
                the bounding boxe
                
            image (numpy.ndarray):
                the image we want 
                
        Returns:
            the image centered 
            
        Raise:
        -----
            - if no objects in bounding box
            - if bounding box is not a list
    
    """
    if len(bounding_box) == 0:
        raise TypeError(f"no bounding box found: {bounding_box}")
    
    if not isinstance(bounding_box, list):
        raise TypeError(f"wrong type object given, expected list found : {type(bounding_box).__name__}")
    
    if not isinstance(image, numpy.ndarray):
        raise TypeError(f"wrong type object given, expected numpy.ndarray found : {type(image).__name__}")
    
    x_min, y_min, x_max, y_max =  bounding_box
    centered_image = image[int(y_min) : int(y_max), int(x_min) : int(x_max)]
    return centered_image

    
def detect_label_image(model: YOLO, image: np.ndarray, specified_label: int = 0):
    """Detect one or several specified label on an image 

        Args:
            model (YOLO): 
                YOLO model
                
            image (np.ndarray): 
                image
                
            specified_label (int):
                0 by default for YOLO model for person label
                
        Returns:
            List(list) : 
                list of bounding boxes corresponding to person label

    """
    bounding_boxes, labels = extract_boxes_yolovn(model, image)
    index_person = [i for i, value in enumerate(labels) if value == specified_label]
    
    return [bounding_boxes[i] for i in index_person]

def draw_rounded_rectangle(image, top_left, bottom_right, color, thickness, corner_radius):
    """
        Draws a rectangle with rounded corners around the detected object.

        Args:
            image (numpy.ndarray): 
                The input image on which to draw.
                
            top_left (tuple): 
                Coordinates of the top-left corner of the bounding box.
                
            bottom_right (tuple): 
                Coordinates of the bottom-right corner of the bounding box.
                
            color (tuple): 
                The color of the rectangle (BGR format).
                
            thickness (int):    
                Thickness of the lines for the rectangle.
                
            corner_radius (int): 
                Radius for rounding the corners.
    """

    # Calculate corner points
    x_min, y_min = top_left
    x_max, y_max = bottom_right
    
    
    
    #proportion
    large = abs(x_max - x_min)
    long = abs(y_max - y_min)

    # Ensure the radius is not too large
    corner_radius = min(corner_radius, (x_max - x_min) // 2, (y_max - y_min) // 2)

    # Draw straight lines for the sides
    # cv2.line(image, (x_min + corner_radius, y_min), (x_max - corner_radius, y_min), color, thickness)
    # cv2.line(image, (x_min + corner_radius, y_max), (x_max - corner_radius, y_max), color, thickness)
    # cv2.line(image, (x_min, y_min + corner_radius), (x_min, y_max - corner_radius), color, thickness)
    # cv2.line(image, (x_max, y_min + corner_radius), (x_max, y_max - corner_radius), color, thickness)
    
    # Draw short lines at each corner (horizontal and vertical segments)
    
    # Top-left corner
    cv2.line(image, (x_min + corner_radius, y_min), (int(large*0.2) + x_min + corner_radius, y_min), color, thickness)  # Horizontal line
    cv2.line(image, (int((0.8*large) + x_min - corner_radius), y_min), (x_max - corner_radius, y_min), color, thickness)  # Horizontal line
    
    cv2.line(image, (x_min + corner_radius, y_max), (int(large*0.2) + x_min, y_max), color, thickness)  # Horizontal line
    cv2.line(image, (int((0.8*large) + x_min), y_max), (x_max - corner_radius, y_max), color, thickness)  # Horizontal line
    
    cv2.line(image, (x_min, y_min + corner_radius), (x_min, y_min + corner_radius + int(long*0.2)), color, thickness)
    cv2.line(image, (x_min, y_min + int(long*0.8)), (x_min, y_max - corner_radius), color, thickness)
    
    cv2.line(image, (x_max, y_min + corner_radius), (x_max, y_min + corner_radius + int(long*0.2)), color, thickness)
    cv2.line(image, (x_max, y_min + int(long*0.8)), (x_max, y_max - corner_radius), color, thickness)

    # Draw the rounded corners
    cv2.ellipse(image, (x_min + corner_radius, y_min + corner_radius), (corner_radius, corner_radius), 180, 0, 90, color, thickness)
    cv2.ellipse(image, (x_max - corner_radius, y_min + corner_radius), (corner_radius, corner_radius), 270, 0, 90, color, thickness)
    cv2.ellipse(image, (x_min + corner_radius, y_max - corner_radius), (corner_radius, corner_radius), 90, 0, 90, color, thickness)
    cv2.ellipse(image, (x_max - corner_radius, y_max - corner_radius), (corner_radius, corner_radius), 0, 0, 90, color, thickness)

    
def display_objects_image(img_cv2: numpy.ndarray, bounding_boxes: List[List[float]])->numpy.ndarray:
    """
        This function takes an image and try to detect object on it

        Args:
            img_cv2 (np.numpy): 
                the image we want to detect objects on, read by cv2
                
            bounding_boxes (List[List[float]]):
                List of list of coordinates
                
        Returns:
            image_illustrated (numpy.ndarray):
                the image with rectangle on detected object
    """
    
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    # adapted to cv2.imread
    img_illustrated =  np.clip(img_rgb * 0.4, 0, 255).astype(np.uint8)
    
    for b in bounding_boxes:
        x_min, y_min, x_max, y_max =  b
        img_illustrated[int(y_min) : int(y_max), int(x_min) : int(x_max)] = bounding_yolovn(b, img_rgb)
        # img_illustrated = cv2.rectangle(img_illustrated, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 255, 0), 5)
        
        draw_rounded_rectangle(
            img_illustrated, 
            (int(x_min), int(y_min)), 
            (int(x_max), int(y_max)), 
            color=(200, 0, 0),  # Cyan color (like Google Lens)
            thickness=5,
            corner_radius=20
        )
    
    #plt.imshow(img_illustrated)
    return img_illustrated