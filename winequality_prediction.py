# =============== Data Loading ================

# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

# Loading up the data
df_wine = pd.read_csv('winequality-red.csv')
df_wine.head()

# =============== Exploratory Data Analysis ================


# Missing Values
df_wine.info()
df_wine.isnull().sum()
df_wine.describe()

# Outliers
sns.boxplot(x=df_wine['fixed acidity'])
sns.boxplot(x=df_wine['volatile acidity'])
sns.boxplot(x=df_wine['citric acid'])

# Univariate EDA
df_wine.hist(bins=50, figsize=(20,15))
plt.show()

# Multivariate EDA
plt.figure(figsize=(10,10))
sns.barplot(x='quality', y='fixed acidity', data=df_wine)

plt.figure(figsize=(10,10))
sns.barplot(x='quality', y='volatile acidity', data=df_wine)

plt.figure(figsize=(10,10))
sns.barplot(x='quality', y='citric acid', data=df_wine)

plt.figure(figsize=(8,8))
corr = df_wine.corr()
sns.heatmap(corr, cbar=True, square=True, fmt='.1f', annot=True, cmap='Reds')

# =============== Data Preparation ================

# Label Encoder
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
df_wine['quality'] = pd.cut(df_wine['quality'], bins=bins, labels=group_names)
df_wine.head()

le = LabelEncoder()
df_wine['quality'] = le.fit_transform(df_wine['quality'])
df_wine.head()

sns.countplot(df_wine['quality'])

df_wine['quality'].value_counts()

# Splitting Data
X = df_wine.drop(["quality"], axis =1)
y = df_wine["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Jumlah sample pada dataset: {len(X)}')
print(f'Jumlah sample pada data train: {len(X_train)}')
print(f'Jumlah sample pada data test: {len(X_test)}')

# =============== Modelling ================

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

knn_acc = accuracy_score(y_test, knn.predict(X_test))
knn_cv = cross_val_score(knn, X, y, cv=10).mean()

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

rf_acc = accuracy_score(y_test, rf.predict(X_test))
rf_cv = cross_val_score(rf, X, y, cv=10).mean()

# Support Vector Machine
svm = SVC()
svm.fit(X_train,y_train)

svm_acc = accuracy_score(y_test, svm.predict(X_test))
svm_cv = cross_val_score(svm, X, y, cv=10).mean()

# LightGBM
lgbm = LGBMClassifier()
lgbm.fit(X_train,y_train)

lgbm_acc = accuracy_score(y_test, lgbm.predict(X_test))
lgbm_cv = cross_val_score(lgbm, X, y, cv=10).mean()

# =============== Evaluation ================

results = {
    'Model': ['K-Nearest Neighbors', 'Random Forest', 'SVM', 'LightGBM'],
    'Accuracy': [knn_acc, rf_acc, svm_acc, lgbm_acc],
    'CV Score': [knn_cv, rf_cv, svm_cv, lgbm_cv],
}

result_model = pd.DataFrame(results)

result_model

# =============== Prediction ================

lgbm.predict([[7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]])