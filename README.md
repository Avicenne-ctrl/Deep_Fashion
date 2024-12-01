# Deep Fashion API

## Overview
Ever found yourself staring at your closet, unsure how to style your clothes? Many garments go unused simply because we lack inspiration on how to wear them. The Deep Fashion Project is here to solve this problem by providing creative outfit ideas tailored to your wardrobe. With this API, you can:

- Search for outfit suggestions based on specific clothing items.
- Find similar outfits by uploading an image for reference.


## Technologies Used
- YOLOv8: Utilized for detecting people in images with state-of-the-art object detection capabilities.
- Roboflow Open Source Model: Used for detailed clothing item detection directly on a person.
- Nearest Neighbors Algorithm: Implemented to identify similar outfits efficiently by comparing features.

## How to use this repository :

### Copy the app repository  
`git clone https://github.com/Avicenne-ctrl/Deep_Fashion.git`  

### Install dependencies  
`pip install -r requirements.txt`  

### Specify your RoboFlow Token API  
`export ROBOFLOW_TOKEN=your_token_roboflow`   

### Start the app.py  
`python app.py`   

## How to Modify the Image Dataset  
- The app uses a default dataset of images stored in static/images/. You can add your own images to this folder to expand the dataset.
- If you update the image dataset, ensure you follow all steps in the create_data_clothes.ipynb notebook to regenerate the clothes_data.csv file, which is essential for proper functionality.  

## How to Change the Detection Model  
- The models used for detection are specified in the config.ini file.
- To update the model, modify the relevant configuration directly in the config.ini file.

## Limitations  
- Model Precision: The accuracy of clothing detection relies heavily on the quality of the RoboFlow model. If the model lacks precision, the results may be suboptimal.
- Color Detection: The app detects the predominant color in an image, which can sometimes be inaccurate. For instance, if the background is the dominant element, the detected color may not correspond to the clothing item itself.

## Script description :

| **File Name**              | **Description**                                                                                              | **Main Function**          | **Parameters**                                                                                                         | **Outputs**                                                                                              |
|----------------------------|----------------------------------------------------------------------------------------------------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `extract_clothes_roboflow.py` | Extracts clothes detected on a person image using a Roboflow model.                                       | `detect_clothes_on_person` | - `model (Roboflow)`: Roboflow pretrained model. <br> - `person_image (np.ndarray)`: Image of the person. <br> - `confidence (float, optional)`: Confidence threshold for accuracy (default: 0.3). | `Dict[str, List[float]]`: Dictionary with clothing names as keys and their bounding boxes as values. |
| `extract_main_color.py`     | Extracts the main color from an image.                                                                    | `get_name_rgb_color`       | - `rgb (np.array)`: RGB array of the color. <br> - `color_database (pd.DataFrame)`: Color database with columns `["color", "R", "G", "B"]`. <br> - `display_color (bool, optional)`: Whether to display the color (default: False). | `str`: Detected color name.                                                                              |
| `extract_person_yolo.py`    | Detects persons in an image using a YOLO model.                                                          | `detect_label_image`       | - `model (YOLO)`: YOLO pretrained model. <br> - `image (np.ndarray)`: Input image. <br> - `specified_label (int)`: Label ID for person detection (default: 0). | `List[list]`: List of bounding boxes corresponding to detected persons.                            |
| `main_utilities.py`         | Creates a CSV dataset linking clothing descriptions to images.                                           | `detect_outfit_color_batch`| - `list_images (List[np.ndarray])`: List of images loaded with `plt.imread()`. <br> - `list_image_names (List[np.ndarray])`: Names of the images. <br> - `model_detect_person (YOLO)`: YOLO pretrained model. <br> - `model_detect_clothes (Roboflow)`: Roboflow pretrained model. <br> - `data_color (pd.DataFrame)`: CSV containing color data (columns: R, G, B). | `pd.DataFrame`: CSV dataset with image information, including clothes and associated colors.                      |
| `search_engine_img.py`      | Implements a search engine for outfits by analyzing an image and retrieving similar styles using nearest neighbors. | `main_search_img`          | - `img_path (str)`: Path to the input image. <br> - `data_color (pd.DataFrame)`: Color dataset CSV. <br> - `neighbors_data (pd.DataFrame)`: Clothes dataset CSV. <br> - `nb_similar (int)`: Number of similar nearest neighbors. <br> - `list_index (List[str | int])`: List of image indices. | List[str | int]: Indices of images similar to the input.                                                  |
| `search_engine.py`          | Finds nearest neighbors based on specific features and attributes.                                        | `get_search_by_knn`        | - `neighbors_queries (Dict[str, str])`: Query dictionary with key-value pairs representing column names and values. <br> - `nb_similarities (int)`: Number of similar objects. <br> - `neighbors_data (pd.DataFrame)`: Data for neighbor search. <br> - `list_index (List[str | int], optional)`: Specific indices if needed. | List[str | int]: Indices of similar objects.                                                             |
