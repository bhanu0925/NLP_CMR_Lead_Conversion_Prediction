import pandas as pd
import nltk
import re
import string
from wordsegment import load, segment
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import contractions
import src.config as config
nltk.download()
nltk.download('punkt')
nltk.download('wordnet')


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix,classification_report

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




def lower_casing_text(text):
    
    """
    The function will convert text into lower case.
    
    arguments:
         input_text: "text" of type "String".
         
    return:
         value: text in lowercase
         
    Example:
    Input : The World is Full of Surprises!
    Output : the world is full of surprises!
    
    """
    # Convert text to lower case
    # lower() - It converts all upperase letter of given string to lowercase.
    text = text.lower()
    return text

def remove_puctuation(text):
    """
        The function will replace punctuation with a space.

        arguments:
             input_text: "text" of type "String".

        return:
             value: text with no punctuations

        Example:
        Input : The World is Full of Pan%tu#at!io#n@s!
        Output : the world is full of Pantuations

        """
    text = text.replace(':',' ')  
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text

def remove_digits_with_words(text):
    
  
    """
    The function will remove digits with traiing or preceedign text.

    arguments:
        input_text: "text" of type "String".

    return:
        value: text with no digits with texts

    Example:
    Input : The World is Full of num234num, 9am
    Output : the world is full of 

    """

    text = re.sub('\w*\d\w*', '', text) #digits with trailing or preceeding text
    return text


def expand_shortend(text,shortend_mapping = config.SHORTEND_MAP):
    """expand shortened words to the actual form.
       e.g. don't -> do not
            tmng  -> timing

       arguments:
            input_text: "text" of type "String".

       return:
            value: Text with expanded form of shorthened words.

       Example: 
       Input : ain't, aren't, can't, cause, can't've
       Output :  is not, are not, cannot, because, cannot have 

     """
    # tokenizeing the text into tokens
    list_of_tokens = text.split(' ')

    # check for the given token in the shortend map, if found, replaces with the mapping value

    temp_lst = []
    for word in list_of_tokens:

        # check the token is in the shortend mapping dict
        if word in config.SHORTEND_MAP:        
            temp_lst.append(word.replace(word,config.SHORTEND_MAP[word]))
        else:
            temp_lst.append(word)

    #converting list of tokens to string.
    string_of_tokens = ' '.join(str(s) for s in temp_lst)
    return string_of_tokens

def expand_contractions(text):
    """expand shortened words to the actual form.
       e.g. don't -> do not
            tmng  -> timing

       arguments:
            input_text: "text" of type "String".

       return:
            value: Text with expanded form of shorthened words.

       Example: 
       Input : ain't, aren't, can't, cause, can't've
       Output :  is not, are not, can not, because, cannot have """
    text = contractions.fix(text)
    return text


def word_segmentation(text,tech_list = config.tech):
    
    """expands concatenated words 
       e.g. moredetails -> more details
            callyou  -> call you

       arguments:
            input_text: "text" of type "String".

       return:
            value: Text with expanded form of concatenated words.

       Example: 
       Input : please sharedetails of the classtimings
       Output : please share details of the class timings

    """
    load()
    text_lst = []
    list_of_tokens = text.split(' ')
    for word in list_of_tokens:
        if word in tech_list:
            text_lst.append(word)
        else:
        #if two words are cancatenated, its seperates the words
            text_lst.extend(segment(word))
    #converting list of tokens to string.
    string_of_tokens = ' '.join(str(s) for s in text_lst)
    return string_of_tokens


def spelling_correction(text):
    '''
    This function will correct spellings.

    arguments:
         input_text: "text" of type "String".

    return:
        value: Text after corrected spellings.

    Example: 
    Input : This is Oberois from Dlhi who came heree to studdy.
    Output : This is Oberoi from Delhi who came here to study.

    '''
    corrected_text= str(TextBlob(text).correct())
    return corrected_text

def custom_stopword_removal(text,cust_stopwords = config.CUSTOM_STOPWORDS, tech_list =config.tech,remove_small_tokens = True, min_len =1):
    """This function will remove stopwords which doesn't add much meaning to a sentence 
       & they can be remove safely without comprimising meaning of the sentence.

    arguments:
         input_text: "text" of type "String".

    return:
        value: Text after omitted all stopwords.

    Example: 
    Input : This is Kajal from delhi who came here to study.
    Output : ["'This", 'Kajal', 'delhi', 'came', 'study', '.', "'"]
    """
       
    
    text_lst = []
    list_of_tokens = text.split(' ')
    for word in list_of_tokens:
        # if word not in custom stopword list and not in tech
        if word not in cust_stopwords:
            # if remove small words
            if remove_small_tokens:                
                #if word length is not min lenth
                if len(word) <= min_len:
                    if word in config.do_not_remove or word in tech_list:
                        text_lst.append(word)                    
                else:
                    text_lst.append(word)

    return ' '.join(text_lst)


def remove_non_English_words(text,tech_list = config.tech):
    
    only_english = list(set(nltk.corpus.words.words()))
    text_lst = []
    for word in word_tokenize(text):
        if word in only_english:
            text_lst.append(word)
        elif word in tech_list:
            text_lst.append(word) 
        
    text = ' '.join(text_lst)
    return text

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()    
    text = [lemmatizer.lemmatize(word) for word in word_tokenize(text)]
    text = [lemmatizer.lemmatize(word,'v') for word in text]
    return ' '.join(text)

def data_cleaning(text):
    
    ## lowercase
    text = lower_casing_text(text)

    ## remove puctuation
    text = remove_puctuation(text)

    ## remove digits with trailing and preceeding text - sdf243abvc, 9am
    text = remove_digits_with_words(text)

    #expand contractions
    text = expand_contractions(text)

    ## word segmentations - > shareddetails -> shared details
    text = word_segmentation(text)

    ## Normalize words
    text = expand_shortend(text,shortend_mapping = config.SHORTEND_MAP)

    ##spelling correction
    text = spelling_correction(text)

    ## scustom stopword removal 
    text = custom_stopword_removal(text)
    
    ## Lemmetization
    text = lemmatize(text)

    ## 
    return text


