#Unsupervised Learning : Principal Component Analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names']) #converting data into Data Frame

from sklearn.preprocessing import StandardScaler #Standardizing data by using Tranformation
scaler = StandardScaler()
scaler.fit(df)
scaled_data=scaler.transform(df)

#Applying PCA to reduce dimentionality of data from 30 to 2
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)


print('scaled data dimensionality:')
print(scaled_data.shape)
print('PCA data dimensionality:')
print(x_pca.shape)

#Plotting the 2 components of data
plt.figure(figsize=(10,4))
plt.scatter(x_pca[:,0],x_pca[:,1], c=cancer['target'], cmap='plasma')
plt.xlabel('1st PC')
plt.ylabel('2nd PC')
plt.show()

#Understanding all features wrt these 2 components
df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])
plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma')
plt.show()