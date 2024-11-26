import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Optional

from ultralytics import YOLO
from roboflow import Roboflow
import roboflow
import torch

import scripts.search_engine as se
import scripts.main_utilities as mu
import configparser

# Read config file
config = configparser.ConfigParser()
config.read('config.ini')

YOLO_MODEL          = config["MODEL"]["YOLO_MODEL"]
ROBOFLOW_MODEL      = config["MODEL"]["ROBOFLOW_MODEL"]

# Load model
device = torch.device('cpu')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODEL_DETECT_OBJ    = YOLO(YOLO_MODEL)
MODEL_DETECT_OBJ.to(device) 

# Connect to Roboflow
roboflow.login()
rf = Roboflow()

# Load pretrained model for clothes recognition
project = rf.workspace().project(ROBOFLOW_MODEL)
fashion = project.version(4).model


def similar_outfit_for_image(neighbors_queries: pd.DataFrame, neighbors_data: pd.DataFrame, nb_similar:int, list_index: List[str | int]):
    """search for similar outfit given a dataframe extraction of an image

    Args:
        neighbors_queries (pd.DataFrame): 
            the dataframe of the images data
            
        neighbors_data (pd.DataFrame): 
            the dt-ataframe we want to find similar objects
            
        list_index (str | int):
            list of index to get the corresponding results
    """
    results = []

    for detected in neighbors_queries.values:
        dict_data = {}
        for i in range(len(detected)):
            dict_data[neighbors_queries.columns[i]] = detected[i]
            
        result = se.get_search_by_knn(dict_data, nb_similar, neighbors_data, list_index)
        
        results += result
        
    return results


def search_engine_img(model_object: YOLO, 
                      model_outfit: Roboflow, 
                      img: np.ndarray, 
                      data_color: pd.DataFrame, 
                      neighbors_data: pd.DataFrame,
                      nb_similar: int, 
                      list_index : List[str | int]):
    """_summary_

    Args:
        model_object (YOLO): _description_
        model_outfit (Roboflow): _description_
        img (np.ndarray): _description_
        data_color (pd.DataFrame): _description_
        neighbors_data (pd.DataFrame): _description_
        nb_similar (int): _description_
        list_index (List[str  |  int]): _description_

    Returns:
        _type_: _description_
    """
    
    outfit_detected = mu.detect_person_outfit_image(model_object, model_outfit, img, data_color)
            
    results = similar_outfit_for_image(outfit_detected, neighbors_data, nb_similar, list_index)
    
    return results

def main_search_img(img_path: str, data_color: pd.DataFrame, neighbors_data: pd.DataFrame, nb_similar: int, list_index : List[str | int]):
    """_summary_

    Args:
        img_path (str): _description_
        data_color (pd.DataFrame): _description_
        neighbors_data (pd.DataFrame): _description_
        nb_similar (int): _description_
        list_index (List[str  |  int]): _description_

    Returns:
        _type_: _description_
    """
    
    img = plt.imread(img_path)
    results = search_engine_img(MODEL_DETECT_OBJ, fashion, img, data_color, neighbors_data, nb_similar, list_index)
    
    return results