from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics

from sklearn.model_selection import GridSearchCV

import imblearn
from imblearn.over_sampling import SMOTE
from numpy import mean