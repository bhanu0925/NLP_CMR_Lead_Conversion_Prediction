
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from src.utils import load_model
from src.datapreprocessing import data_cleaning
import pandas as pd


def ordinal_encoder(input_val, feats): 
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats
    feat_dict = dict(zip(feat_key, feat_val))
    value = feat_dict[input_val]
    return value


def get_prediction(text,location,model):
    """
    Predict the class of a given data point.
    """
    tfidfVec = load_model(r"model/tfidfVec")
    ohe = load_model(r"model/onehotencoder")
    text = data_cleaning(text)
    
    
    df_text_tfidf = pd.DataFrame(tfidfVec.transform([text]).todense(),columns=tfidfVec.get_feature_names_out())
    loc_ohe = pd.DataFrame(ohe.transform([[location]]).toarray(),columns=ohe.get_feature_names_out())
    df_test_pred = pd.concat((df_text_tfidf,loc_ohe),axis=1)
   
    pred = model.predict(df_test_pred)

    if pred[0] == 0:
        return "Lead will not convert"
    else:
        return "Lead will convert"
    
