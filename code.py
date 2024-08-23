import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
df = pd.read_csv('C:/Users/super/Downloads/heart.csv')

# Exploratory Data Analysis (EDA)
print(df.head())
print(df.info())
print(df.describe())

# Correlation matrix
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(16, 16))
sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()

# Target distribution
sns.set_style('whitegrid')
sns.countplot(x='target', data=df, palette='RdBu_r')
plt.show()

# Data preprocessing
dataset = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Handle missing values (if any)
# dataset.fillna(method='ffill', inplace=True)  # or other imputation methods

# Feature scaling
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

# Split data into features and target
X = dataset.drop('target', axis=1)
y = dataset['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Classifier with hyperparameter tuning
knn_params = {'n_neighbors': range(1, 21), 'metric': ['euclidean', 'manhattan']}
knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, knn_params, cv=5)
grid_search.fit(X_train, y_train)
best_knn = grid_search.best_estimator_
knn_pred = best_knn.predict(X_test)

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)

# Model evaluation
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

print("KNN Classification Report:\n", classification_report(y_test, knn_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))

print("KNN Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
from sklearn.model_selection import cross_val_score
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score=cross_val_score(knn_classifier,X,y,cv=10)
    knn_scores.append(score.mean())
plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')
plt.show()
knn_classifier = KNeighborsClassifier(n_neighbors = 12)
score=cross_val_score(knn_classifier,X,y,cv=10)
score.mean()
