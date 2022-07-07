from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd



def main():
    df = pd.read_csv(r'C:\Users\karol\Project-ML\src\data\data_ML\train_data.csv', header=None)
    y = pd.read_csv(r'C:\Users\karol\Project-ML\src\data\data_ML\train_labels.csv', header=None)

    scaler = StandardScaler()
    df_std = scaler.fit_transform(df)

    tsne = TSNE(n_components=2)
    pca = PCA(n_components=0.99)

    xpca = pca.fit_transform(df_std)
    pca_results = tsne.fit_transform(xpca)
    
    pca_tsne_results = tsne.fit_transform(pca_results)
    

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=pca_tsne_results,
        x=pca_tsne_results[:,0], 
        y=pca_tsne_results[:,1],
        hue=y[0],
        palette=sns.color_palette("Paired", as_cmap=True),
        alpha=0.5
    )
    plt.show()

if __name__ == "__main__":
    main()