

# # NLP_CMR_Lead_Conversion_Prediction

## **Classification project Description**:
Project Description:Understanding customer patterns is one of the important activities in
every business, based on customer pattern and customer status our next step was majorly
planned in every business process. Goal of the project was to identify the lead conversion possibilities based on location and conversation between buissiness executive and candidates. The proposed solution where achieved using NLP techniques and different machine learning algorithms

## **Problem Statement**
Based on location, status and type of business executive identify whether the lead is going to be converted on not.
## **App Demo**
<div align='center'>
<img src="https://github.com/bhanu0925/NLP_CMR_Lead_Conversion_Prediction/blob/main/RTA.gif">
</div>

**Steps followed**

**Data Pro-processing/EDA

- Since this was text data lot of preprocessing has been done

1. Lower casing the text data
2. Removal of panctuations
3. Removing words with digits
4. EXpanding shortend words
5. Expanding contractions
6. word segmentations
7. Spelling corrections
8. Custom Stop words Removal
9. Removing non english words
10. Lemmatization

**Feature engineeting and Text representations
This problem staements depands to predict outcome based on two features, categorical and text.
FEature engineering is done in two ways and tested

1. Features 1 - Location information was added along with text information and text vectarization is done using Bag og words of unigrams
3. Features 2 - Location information was one hot endoded and Text information is text vectarized separetly using term frequency-inverse document frequency

**Imbalance data Handling

1. SMOTE is used for data balancing

**Modelling

-1 Machine learning algorithms like 
 - Logistic Regression algorithm
 - Naive Bayes algorithm
 - Support vector machine
 - Randomforest classification algorithm
 - XXBoost 

 where considered for experimets.
 SVC outperformed with better metrics. This application is using SVC at the backend

**Model Evaluation

 - Precision and Recall were considered as evaluation metrics 

**App design and Deployment

 - This all is developed using streamlit
 - Deployed on Heroku


