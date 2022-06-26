## EDA
First we had to look into our data. We did some basic statistic.
There was no missing values nor duplicates. We used Z-score to determinate
outliners. Z-score is just the number of standard deviations away from the 
mean that a certain data point is. The method removed about 2800 columns with
outliners. 

Next we checked out the correlation coefficients between variables using correlation
matrix. We threw out the columns with correlation greater than 0.8.

Time for PCA. First we scaled the features using StandardScaler to make the optimal
performance of machine learning algorithms. Then we visualized it by plotting 2 
dimensional data. The reduced 2-dimensional data still contains 0.2% of the variance 
of the original data.

T-SNE. This stochastic method is used primarily for the exploration and visualization 
of multidimensional data.

We saved our data using joblib for further analyze.

Moving on to splitting our data to train and test. We used train_test_split with test 
size of 0.2. 

Implementing classifications, we did 4:
+ DummyClassifier
+ LogisticRegression
+ KNClassifier
+ Random Forest Classifier