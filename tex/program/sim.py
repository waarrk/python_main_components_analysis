from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import numpy as np

# Irisデータセットの読み込み
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

# データの表示
print(iris_df.head())

# Irisデータセットの読み込み
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

# 特徴量の標準化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(iris_df.iloc[:, :-1])

# 主成分分析の実施
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# 主成分得点をデータフレームに追加
pca_df = pd.DataFrame(pca_result, columns=[
                      'Principal Component 1', 'Principal Component 2'])
pca_df['species'] = iris_df['species']

# 品種のラベルを文字列に変換
species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
pca_df['species'] = pca_df['species'].map(species_map)

# 主成分得点のプロット
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x='Principal Component 1', y='Principal Component 2',
    hue='species',
    palette=sns.color_palette("hsv", 3),
    data=pca_df,
    legend='full'
)
plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Flower Species')
plt.show()

# 分散説明率の計算
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# 再構成誤差の計算
pca_back = pca.inverse_transform(pca_result)
reconstruction_error = mean_squared_error(scaled_data, pca_back)

# 評価結果の表示
print(f'分散説明率（各主成分）: {explained_variance_ratio}')
print(f'累積分散説明率: {cumulative_variance_ratio}')
print(f'再構成誤差: {reconstruction_error}')
