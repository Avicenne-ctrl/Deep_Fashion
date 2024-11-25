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


### Extract color 

def preprocess_image(image):
    """
        Prepare the image for color recognition process
        
        Args:
            image (np.ndarray):
            
        Returns:
            np.ndarray
    """
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    return pixels

def get_dominant_cluster(pixels, k=1):
    """We create cluster for similar pixel on the image
        then we select the dominant cluster

        Args:
            pixels (_type_): _description_
            k (int, optional): 
                Number of cluster. Defaults to 1.

        Returns:
            np.ndarray: tuple of RGB color
    """
    # create pixel cluster
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    labels = kmeans.labels_

    # Count label frequency
    label_counts = Counter(labels)

    # Sort cluster to get the dominant first
    sorted_clusters = sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True)

    # Get the k dominant color, here only 1
    dominant_colors = kmeans.cluster_centers_[sorted_clusters[:k]].astype(int)
    
    return dominant_colors

def get_dominant_color(image: np.ndarray):
    """Get the dominant color on an image

        Args:
            image (np.ndarray): 
                
        Returns:
            np.array: 
                the RGB np.ndarray of the dominant color
    """
    
    preprocessed_image = preprocess_image(image)
    dominant_color_rgb = get_dominant_cluster(preprocessed_image, k=1)[0]

    return dominant_color_rgb

def get_name_rgb_color(rgb: np.array, color_database: pd.DataFrame, display_color: bool = False):
    """return the color name given an RGB np.array

        Args:
            rgb (np.array): 
                the RGB np.array
                
            color_database (pd.DataFrame): 
                the color database, columns required -> ["color", "R", "G", "B"]
                
            display_color (bool):
                if we want to display the color. False by default

        Returns:
            str: color name
            
        Example:
        --------
            >>> get_name_rgb_color(rgb = [255, 0, 0], color_database = pd.DataFrame({"color": ["Red", "Green"],
                                                                                        "R" : [255, 0],
                                                                                        "G" : [0, 255],
                                                                                        "B" : [0, 0]}))
        >>> "Red"
    """

    
    X = color_database.copy()
    
    label_color = X.pop('color').tolist()

    knn = NearestNeighbors(algorithm = "brute", metric = "euclidean")
    knn = knn.fit(X)
    
    indices = list(knn.kneighbors([rgb], 1, return_distance = False))
    
    if display_color:
        image = [[rgb for _ in range(100)] for _ in range(100)]

        # Afficher l'image
        plt.imshow(image)
        plt.axis('off')  # Masquer les axes
        plt.show()
    
    return label_color[int(indices[0])]