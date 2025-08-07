import os, sys 
import numpy as np
## import dill 
import pickle 

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise networkscurityException(e,sys) from e

def write_yaml_file(file_path: str, content: object,replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(content, yaml_file)
    except Exception as e:
        raise networkscurityException(e, sys) from e
    
def save_numpy_array_data(file_path: str, array: np.array):
    '''
    Save numpy array data to a file.
    file_path: str: Path to save the numpy array.
    array: np.array: Numpy array to be saved.
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise networkscurityException(e, sys) from e

