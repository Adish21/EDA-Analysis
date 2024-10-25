import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('cancer.csv')
data.head()
data.tail()
data.shape
data.describe()
data.columns
data.nunique()

data['diagnosis(1=m, 0=b)'].nunique()
data.isnull().sum()

cancer = data.drop(['perimeter_mean','smoothness_mean','compactness_mean','fractal_dimension_mean'], axis=1)
cancer.head()

corelation = cancer.corr()
sns.heatmap(corelation, xticklabels = corelation.columns, yticklabels = corelation.columns
            , annot = True)

sns.pairplot(cancer)

sns.distplot(cancer['area_mean'],bins = 20)

#clustering of the data
print(cancer.shape)
print(cancer.head())
print(cancer.info())

# Adding features of the data as the columns
features = ['diagnosis(1=m, 0=b)', 'radius_mean', 'area_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean']

# Number of clusters adding the data
k = 4
kmeans = KMeans(n_clusters=k)
cluster_labels = kmeans.fit_predict(data[features])

#labels
silhouette_avg = silhouette_score(data[features], cluster_labels)
print("Silhouette Score:", silhouette_avg)

#Printing clusters
data['Cancer'] = cluster_labels
sns.scatterplot(data=data, x='diagnosis(1=m, 0=b)', y='radius_mean', hue='Cancer')
plt.show()

data['Cancer'] = cluster_labels
sns.scatterplot(data=data, x='area_mean', y='concavity_mean', hue='Cancer')
plt.show()

data['Cancer'] = cluster_labels
sns.scatterplot(data=data, x='concave points_mean', y='symmetry_mean', hue='Cancer')
plt.show()
