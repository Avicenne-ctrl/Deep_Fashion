from flask import Flask, render_template, request

import scripts.search_engine as se
import scripts.search_engine_img as sei
import scripts.main_utilities as mu

import pandas as pd
from collections import Counter
import configparser

# Read config file
config = configparser.ConfigParser()
config.read('config.ini')

PATH_IMAGES_FOLDER  = config["PATH"]["PATH_IMAGES_FOLDER"]
PATH_DATA_COLOR     = config["PATH"]["PATH_DATA_COLOR"]
PATH_SAVE_DATA      = config["PATH"]["PATH_SAVE_DATA"]

YOLO_MODEL          = config["MODEL"]["YOLO_MODEL"]
ROBOFLOW_MODEL      = config["MODEL"]["ROBOFLOW_MODEL"]

EXTENSION_IMG         = ["jpeg", "png", "jpg"]


# Dataset with image path and its clothes, color description
neighbors_data = pd.read_csv(PATH_SAVE_DATA)
data_color     = pd.read_csv(PATH_DATA_COLOR)

# we save id list and drop
image_names    = [PATH_IMAGES_FOLDER + img_name for img_name in neighbors_data["id"]]

# Get the list of all the images in the data folder
path_img_list, img_names = mu.get_extension_folder(PATH_IMAGES_FOLDER, EXTENSION_IMG)

# Get unique value from the dataframe
tops_unique          = neighbors_data["top"].dropna().unique()
bottoms_unique       = neighbors_data["bottom"].dropna().unique()
tops_color_unique    = neighbors_data["top_color"].dropna().unique()
bottoms_color_unique = neighbors_data["bottom_color"].dropna().unique()

print(tops_unique)
print(bottoms_unique)
print(tops_color_unique)
print(bottoms_color_unique)

app = Flask(__name__)
@app.route("/")
@app.route('/', methods=['GET', 'POST'])

def index():
    
    if request.method == 'POST':
        
        # we get the values selected by the user on the web site
        top_color       = request.form.get('couleurHaut')
        bottom_color    = request.form.get('couleurBas')
        top             = request.form.get('haut')
        bottom          = request.form.get('bas')
        quantite        = request.form.get('quantite')
        
        image_query     = request.form.get('filename')
        
        # Nothing selected -> go back to home page
        if len(quantite) == 0 or (len(top_color) + len(bottom_color)+len(top) + len(bottom) == 0) and (image_query is None or image_query == ""):
            print("[INFO] : no valid request")
            return render_template('index.html', imageList = path_img_list, tops= tops_unique, top_color= tops_color_unique, bottom= bottoms_unique, bottom_color= bottoms_color_unique)
        
        # If an image is uploaded -> extract outifit from the images
        if image_query is not None and image_query != "":
            print(f"[INFO] : similarity search given an image {image_query}")

            image_query_path = PATH_IMAGES_FOLDER + image_query
            results = sei.main_search_img(image_query_path, data_color, neighbors_data, quantite, image_names)
            
            non_duplicates_result = list(Counter(results).keys())
            return render_template('index.html', imageList = non_duplicates_result, tops= tops_unique, top_color= tops_color_unique, bottom= bottoms_unique, bottom_color= bottoms_color_unique)
            
        # If not image uplaoded -> use the input data
        if len(quantite) != 0 and (len(top_color) + len(bottom_color)+len(top) + len(bottom) != 0) and (image_query is None or image_query == ""):   
            
            # the input are stored in a dataset: be aware that the keys name must match the columns name of the dataset df
            neighbors_queries = {'bottom': bottom, 'bottom_color': bottom_color,
                                'top': top, 'top_color': top_color}
            
            print(f"[INFO] : similarity search given a query {neighbors_queries}")

            # the main algo which will returns the images for the corresponding input
            resultats = se.get_search_by_knn(neighbors_queries, quantite, neighbors_data, image_names)

            # we display images on the web site
            non_duplicates_result = list(Counter(resultats).keys())
            return render_template('index.html', imageList = non_duplicates_result, tops= tops_unique, top_color= tops_color_unique, bottom= bottoms_unique, bottom_color= bottoms_color_unique)
        
    return render_template('index.html', imageList = path_img_list, tops= tops_unique, top_color= tops_color_unique, bottom= bottoms_unique, bottom_color= bottoms_color_unique)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="5050", threaded=False)
    
