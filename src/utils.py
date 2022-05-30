import requests
import urllib
import pickle
import os

def get_model_from_Gdrive(model_path):
    """
    Function to load model from g drive
    """
    
    try:
        with open(model_path, "rb") as mh:
            rf = pickle.load(mh)
    except:
        print("Cannot fetch model from local downloading from drive")
        if not 'RT_rePickle.joblib' in os.listdir('.'):
            
            url = "https://drive.google.com/u/0/uc?id=1-G8-t3RwR9FukA3S8nabMsY5l5RXCc9B&export=download&confirm=t"
            r = requests.get(url, allow_redirects=True)
            open(r"RT_rePickle.joblib", 'wb').write(r.content)
            del r
        with open(r"RT_rePickle.joblib", "rb") as m:
            print(m)
            rf = pickle.load(m)
    return rf

def save_file(model_path, obj):
    """
    Function to save an object as pickle file
    """
    with open(model_path, 'wb') as f:
        pickle.dump(obj, f)
        
def load_model(model_path):
    """
    Function to load a pickle object
    """
    return pickle.load(open(model_path, "rb"))