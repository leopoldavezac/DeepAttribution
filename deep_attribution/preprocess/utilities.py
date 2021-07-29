from typing import List, Dict

from sklearn.preprocessing import OneHotEncoder

from numpy import ndarray, zeros

def create_categories_for_one_hot_encoding(campaign_nm_to_index: Dict) -> List:

    nb_categories = len(campaign_nm_to_index.keys())
    categories = zeros(nb_categories)

    for k, v in campaign_nm_to_index.items():
        categories[v] = k
    
    return categories


def one_hot_encoding(X: ndarray, categories: List[str]) -> ndarray:

    encoder = OneHotEncoder(categories=categories)
    
    return encoder.fit_transform(X)
