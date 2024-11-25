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


### YOLO extraction

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

### Display images

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

### Combine function

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
    
    clothes = detect_clothes_on_person(model, person_image)
    print(f"[INFO] : {len(clothes)} outfit found")
    
    if len(clothes)>0:
        for key, item in clothes.items():
            
            # Get clothe color
            clothes_img = bounding_yolovn(item, person_image)
            rgb_color   = get_dominant_color(clothes_img)
            color_name  = get_name_rgb_color(rgb_color, database_color)
                
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
    bounding_person = detect_label_image(model_detect_person, image, label_person_yolo)

    print(f"[INFO] : {len(bounding_person)} person found")
    clothes_data = pd.DataFrame()

    for bp in bounding_person:
        person_image = bounding_yolovn(bp, image)
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

