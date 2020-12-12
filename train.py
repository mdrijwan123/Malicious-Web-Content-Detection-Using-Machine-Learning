# Purpose - This file is used to create a classifier and store it in a .pkl file. You can modify the contents of this
# file to create your own version of the classifier.

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
import joblib

sns.set()

# Read the data
data = pd.read_csv('dataset.csv')
data = data.drop(columns=['index'])
plt.figure(figsize=(8, 8))
plt.show()
sns.heatmap(data.corr())
X = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 27, 29]].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n\n ""Random Forest Algorithm Results"" ")
clf = RandomForestClassifier(min_samples_split=7, verbose=True)
clf.fit(X_train, y_train)
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

pred = clf.predict(X_test)
print(classification_report(y_test, pred))
print('The accuracy is:', accuracy_score(y_test, pred))
print(metrics.confusion_matrix(y_test, pred))

# save the model to disk
# filename = 'classifier/finalized_model.sav'
# joblib.dump(clf, filename)
