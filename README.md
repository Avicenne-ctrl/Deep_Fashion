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

## How to modify the image dataset :
- This app is limited by the default dataset of images stored in : `static/images/`. You can add your own images
- If you have updated the dataset of images, you have to follow each step in the `create_data_clothes.ipynb` to update the ``clothes_data.csv`

## How to change model detection :
- the model are specified in the config.ini
- you can change it directly in the config.ini

## Limitations :
- the precision really depend on the quality of the RoboFlow model
- the color detection can be wrong because it only detect the main color on an image. So if the main element is the background, then the color is not the color of the clothe detected
