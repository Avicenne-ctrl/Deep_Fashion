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

import scripts.extract_clothes_roboflow as ecr
import scripts.extract_main_color as emc
import scripts.extract_person_yolo as epy

def get_clothes_and_color(model: Roboflow, person_image: np.ndarray, database_color:pd.DataFrame, confidence: float = 0.3):
    """Detect clothes on an image with associated color
        Need to update hard coded list containing name of clothes depending on top or bottom

        Args:
            model (Roboflow): 
                Roboflow pretrained model
                
            person_image (np.ndarray): 
                the image with the person
                
            database_color (pd.DataFrame):
                the databse of color with color name and RGB value
                
            confidence (float, optional): 
                The confidence value for the accuracy. Defaults to 0.3.
            
        Returns:
            (Dict[str, List[float]]):
                dict with clothe name as a key and corresponding bounding boxe as a value
    """
    
    
    top_items    = ["shirt", "jacket", "t-shirt", "tee shirt", "polar", ""]
    bottom_items = ["pants", "cargo"]
    data_clothes = {"bottom" : np.nan, "bottom_color": np.nan, "top": np.nan, "top_color": np.nan}
    
    clothes = ecr.detect_clothes_on_person(model, person_image)
    print(f"[INFO] : {len(clothes)} outfit found")
    
    if len(clothes)>0:
        for key, item in clothes.items():
            
            # Get clothe color
            clothes_img = epy.bounding_yolovn(item, person_image)
            rgb_color   = emc.get_dominant_color(clothes_img)
            color_name  = emc.get_name_rgb_color(rgb_color, database_color)
                
            if key in top_items:
                data_clothes["top"] = key
                data_clothes["top_color"] = color_name.lower()
                
            else:
                data_clothes["bottom"] = key
                data_clothes["bottom_color"] = color_name.lower()
                
        return data_clothes
    
    else:
        return None

def detect_person_outfit_image(model_detect_person: YOLO, model_detect_clothes: Roboflow, image: np.ndarray, data_color: pd.DataFrame):
    """_summary_

        Args:
            model_detect_person (YOLO): 
                the pretrained model to detect person on image
            
            model_detect_clothes (Roboflow): 
                pretrained model to detect colthes on a person
            
            image (np.ndarray): 
                image we want to detect clothes on person
            
            data_color (pd.DataFrame): 
                the dataframe of color 
                
        Returns:
            pd.DataFrame:
                the dataframe with colthes and color associated
    """
    
    label_person_yolo = 0
    bounding_person = epy.detect_label_image(model_detect_person, image, label_person_yolo)

    print(f"[INFO] : {len(bounding_person)} person found")
    clothes_data = pd.DataFrame()

    for bp in bounding_person:
        person_image = epy.bounding_yolovn(bp, image)
        extract_data = get_clothes_and_color(model_detect_clothes, person_image, data_color)
        
        if extract_data is not None:
            clothes_data = pd.concat([clothes_data, pd.DataFrame([extract_data])])
            
    return clothes_data

def detect_outfit_color_batch(list_images: List[np.ndarray], list_image_names: List[np.ndarray], 
                              model_detect_person: YOLO, model_detect_clothes: Roboflow, data_color: pd.DataFrame):
    """_summary_

    Args:
        list_images (List[np.ndarray]): _description_
        list_image_names (List[np.ndarray]): _description_
        model_detect_person (YOLO): _description_
        model_detect_clothes (Roboflow): _description_
        data_color (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """

    global_data = pd.DataFrame()
    
    for img, name in zip(list_images, list_image_names):
        data_img = detect_person_outfit_image(model_detect_person, model_detect_clothes, img, data_color)
        
        # add image name
        data_img["id"] = [name]*len(data_img)
        
        global_data = pd.concat([global_data, data_img])
        
    return global_data


### get all the file with specific extension

def get_extension_folder(folder_path: str, extensions: list):
    """
        Get all the extension file in a specified folder
        ignoring the sub folder
    

        Args:
            folder_path (str): 
                the path to the folder
                
            extensions (List[str]): 
                list of extension we want

        Returns:
            list: 
                list of the path to the wanted extension file
    """
    file_paths = []
    file_names = []
    
    # Read the folder
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        
        # Check if its the right extension
        for ext in extensions:
            if os.path.isfile(file_path) and file.endswith(ext):
                # Add the path to the list
                file_paths.append(file_path)
                file_names.append(file)
    
    return file_paths, file_names