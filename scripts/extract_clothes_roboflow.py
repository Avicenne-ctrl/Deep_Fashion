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
    
    
### Roboflow model to extract clothes
    
def detect_clothes_on_person(model: Roboflow, person_image: np.ndarray, confidence: float = 0.3):
    """Detect clothes on an image

        Args:
            model (Roboflow): 
                Roboflow pretrained model
                
            person_image (np.ndarray): 
                the image with the person
                
            confidence (float, optional): 
                The confidence value for the accuracy. Defaults to 0.3.
            
        Returns:
            (Dict[str, List[float]]):
                dict with clothe name as a key and corresponding bounding boxe as a value
    """
    
    clothes = {}
    
    result = model.predict(person_image, confidence=confidence).json()
    for item in result["predictions"]:
        width  = item['width']//2
        height = item['height']//2
        x_min  = item['x'] - width
        y_min  = item['y'] - height
        x_max  = item['x'] + width
        y_max  = item['y'] + height
        
        clothes[item["class"]] = [x_min, y_min, x_max, y_max]
        
    return clothes


