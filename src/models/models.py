import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib 
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sympy import prime
warnings.filterwarnings("ignore")

def main():

    X = joblib.load(r'C:\Users\karol\Project-ML\src\features\train_data_2.pkl')
    y = pd.read_csv(r'C:\Users\karol\Project-ML\src\data\data_ML\train_labels.csv', header=None)
    y = y.values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    scaler = StandardScaler(with_std=False)
    X_train = scaler.fit_transform(X_train, y_train)
    X_test = scaler.transform(X_test)

    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)


    model_params = {
        'SVC': {
            'model': SVC(gamma='auto'),
            'params' : {
                'C': [1,10,20],
                'degree': [1,2,3],
                'kernel': ['linear', 'poly', 'rbf'  ]
            }}  
        ,
        'Random': {
            'model': RandomForestClassifier(),
            'params' : {
                'n_estimators': [1,5,10],
                'max_features': [1, 2, 3]
                        }
        },
        'Log': {
            'model': LogisticRegression(solver='liblinear', multi_class='auto'),
            'params': {
                'C': [1,5,10],
                'penalty': ['l1', 'l2']
            }
        },
        'KNClassifier':{
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [1,3,5]
            }}
        
    }




    scores = []

    for model_name, mp in model_params.items():
        clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False, n_jobs=-2)
        clf.fit(X, y)
        scores.append({
             'model': model_name,
             'best_score': clf.best_score_,
             'best_params': clf.best_params_,
             })
        print(scores)

    df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
    print(df)

if __name__ == '__main__':
    main()

