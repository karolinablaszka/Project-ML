import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import joblib 
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

warnings.filterwarnings("ignore")

X = joblib.load('train_data_2.pkl')
y = pd.read_csv(r'src\data\data_ML\train_labels.csv', header=None)
y = y.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_test = scaler.transform(X_test)

smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

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


#rfc_clf = RandomForestClassifier(random_state = 42, class_weight='balanced')
#rfc_clf.fit(X_train, y_train)

#classification(rfc_clf, X_train, y_train, X_test, y_test)

#lo_clf = LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear', multi_class='auto', max_iter=1000)
#lo_clf.fit(X_train, y_train)

#classification(lo_clf, X_train, y_train, X_test, y_test)

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(class_weight='balanced'),
        'params' : {
            'n_estimators': [1,5,10],
            'classifier__max_features': [1, 2, 3],
            "criterion": ["gini", "entropy"]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear', multi_class='auto', class_weight='balanced'),
        'params': {
            'C': [1,5,10]
        }
    },
    'kn_classifier' : {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [1,3,5]
        }
    }
}




scores = []

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False, n_jobs=-1)
    clf.fit(X, y.values.ravel())
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })


df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(df)


