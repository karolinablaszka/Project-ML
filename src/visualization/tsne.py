import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd


def main():

    X = pd.read_csv(r'C:\Users\karol\Project-ML\src\data\data_ML\train_data.csv', header=None)
    y = pd.read_csv(r'C:\Users\karol\Project-ML\src\data\data_ML\train_labels.csv', header=None)

    pca = PCA(n_components=2)
    # fit on all numerical features and reduce dimensionality to two dimensions

    scaler = StandardScaler()
    df_std = scaler.fit_transform(X)

    data_pca = pca.fit_transform(df_std)

    plt.figure(figsize=(10,7))
    sns.scatterplot(x = data_pca[:,0], y = data_pca[:,1], data=data_pca,  hue=y[0], palette=sns.color_palette("Paired", as_cmap=True), linewidth=0.2, alpha=0.9)
    plt.title(f"PCA of numerical features")
    plt.show()

if __name__ == "__main__":
    main()