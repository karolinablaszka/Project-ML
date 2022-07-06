import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import joblib 
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings("ignore")

def main():

    X = joblib.load('train_data_2.pkl')
    y = pd.read_csv(r'src\data\data_ML\train_labels.csv', header=None)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    
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
        
    scaler = StandardScaler(with_std=False)
    X_train = scaler.fit_transform(X_train, y_train)
    X_test = scaler.transform(X_test)

    from imblearn.over_sampling import SMOTE
    sm = SMOTE()
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    y_train_sm.value_counts(normalize=True)

    best_clf= SVC(gamma='auto', C=1, kernel='rbf')    
    best_clf.fit(X_train, y_train)

    classification(best_clf, X_train_sm, y_train_sm, X_test, y_test)
   

if __name__ == '__main__':
    main()
