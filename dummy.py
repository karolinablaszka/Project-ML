import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import joblib 
import warnings
warnings.filterwarnings("ignore")



X = joblib.load('train_data_2.pkl')
y = pd.read_csv(r'src\data\data_ML\train_labels.csv', header=None)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
dummy = dummy_clf.fit(X_train, y_train)
y_pred = dummy_clf.predict(X_test)


def classification(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
        
    #Classification report
    print("CLASSIFICATION REPORT")
    print("------------------------------------------")
    print(classification_report(y_test, y_pred))
    
    
    #Plotting the normalized confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    matrix = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)
    matrix.plot()
    plt.show()
    

classification(dummy_clf, X_train, y_train, X_test, y_test)

y_train.value_counts(normalize=True)
# We clearly have a class imbalance problem
# To address this we can SMOTE the training data and see 
# if training a model with this method would improve our results.

from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
y_train_sm.value_counts(normalize=True)

clf_dummy_sm = DummyClassifier()
clf_dummy_sm.fit(X_train, y_train)

classification(clf_dummy_sm, X_train_sm, y_train_sm, X_test, y_test)