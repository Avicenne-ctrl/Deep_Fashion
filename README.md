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

`git clone `  

### Specify your RoboFlow Token API
`export ROBOFLOW_TOKEN=your_token_roboflow`  

### Start the app.py
`python app.py`  
