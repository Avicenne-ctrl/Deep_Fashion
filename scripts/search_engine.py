from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from typing import List, Dict

def get_knn_similarities(neighbors_data: pd.DataFrame,
                            nb_similarities: int,
                            neighbors_queries: pd.DataFrame,
                            List_index: list,
                            algorithm: str = "brute",
                            metric: str = "euclidean"):
    """
        This fonction takes the critere of search on the dataset, and use KNN
        to return the line of the dataset which corresponde to the request

        Args:
            df_labelize (pd.DataFrame):
                the labelized dataframe where we look for similar case
                
            nb_similarities (int):
                the number of similar cases we want
                
            reference_case_encoded (Dict):
                dict of columns we search similarities with it encoded value corresponding
                
            List_index (list):
                the list of index were we could find the similar case
            
        Returns:
            sol(list):
                list of index corresponding to similar case
    """
    # we initiate the KNN with the labelize dataset with only the columns corresponding to the choices of the user
    knn = NearestNeighbors(algorithm = "brute", metric = "euclidean")
    
    # we get only the column sected for the similarities
    neighbors_data_encode = neighbors_data[neighbors_queries.columns]
    neighbors_queries_encode = neighbors_queries.copy()
        
    # we labelize both dataset and queries
    encode = LabelEncoder()
    for col in neighbors_data_encode.columns:
        neighbors_data_encode[col] = encode.fit_transform(neighbors_data_encode[col])
        neighbors_queries_encode[col] = encode.transform(neighbors_queries_encode[col])
            
    # find similarities
    knn = knn.fit(neighbors_data_encode)
    
    # the most similar line in the dataset
    distance, indices = list(knn.kneighbors(np.array(neighbors_queries_encode.values), nb_similarities, return_distance = True))
    
    sol = [List_index[indice] for indice in indices[0]]
        
    return sol

def remove_dict_empty_value(data_dict: Dict[str, str]):
    """_summary_

    Args:
        data_dict (Dict[str, str]): _description_

    Returns:
        _type_: _description_
    """
    
    dict_transfo = {}
    for key, value in data_dict.items():
        if value != '':
            dict_transfo[key] = value
    return dict_transfo

def get_search_by_knn(neighbors_queries: Dict[str, str], nb_similarities: int, neighbors_data: pd.DataFrame, list_index: List[str | int]):
    """
        This algo takes the input selected by the user on the web site, and return the id of the images that correspond to the input (request)

        Args:
            neighbors_queries (Dict[str, str]): 
                the dict query with key, value -> column_name, value_column
            
            nb_similarities (int): 
                number of similar objects
                
            neighbors_data (pd.DataFrame): 
                the data we will look for neighbors
                
            list_index (List[str  |  int]): 
                if we need to give a specifi index

        Returns:
            List[str | int]: 
                the list of similar index
    """

    # check the non empty values selected by the user
    neighbors_queries_transfo = remove_dict_empty_value(neighbors_queries)
    # we convert into a dataframe
    neighbors_queries_transfo = pd.DataFrame([neighbors_queries_transfo])
    
    # encode user choice
        
    sol = get_knn_similarities(neighbors_data, int(nb_similarities), neighbors_queries_transfo, list_index)

    return sol
