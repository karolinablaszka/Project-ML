import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

def main():

    X = pd.read_csv(r'C:\Users\karol\Project-ML\src\data\data_ML\train_data.csv', header=None)
    y = pd.read_csv(r'C:\Users\karol\Project-ML\src\data\data_ML\train_labels.csv', header=None)
    y = y.values.ravel()
    test_data = pd.read_csv(r'C:\Users\karol\Project-ML\src\data\data_ML\test_data.csv', header=None)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify=y)

    scaler = StandardScaler(with_std=False)
    X_train = scaler.fit_transform(X_train,y_train)

    best_clf= SVC(C=1, degree=2, gamma='auto',
            kernel='poly')
    best_clf.fit(X_train, y_train)
    prediction = best_clf.predict(test_data)
    np.savetxt('predictions.csv', prediction, fmt='%i')

if __name__ == '__main__':
    main()