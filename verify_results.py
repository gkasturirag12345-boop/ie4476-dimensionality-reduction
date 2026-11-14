"""
Supplementary verification of experimental_results.csv.

Re-computes the cumulative explained variance for every tested dimensionality,
re-runs PCA + KNN for 30 components and the 784-D baseline using the same
settings as main.py, and applies a paired McNemar test to the two sets of
predictions. This script only reads files; it does not overwrite any results.
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

K_NEIGHBORS = 5

# Load and preprocess exactly as in main.py
data = np.load('data/mnist.npz')
X_train = data['x_train'].reshape(60000, -1).astype(np.float32) / 255.0
X_test = data['x_test'].reshape(10000, -1).astype(np.float32) / 255.0
y_train = data['y_train'].astype(np.int32)
y_test = data['y_test'].astype(np.int32)

results_df = pd.read_csv('results/experimental_results.csv')

# 1. Error rate must equal 100 - accuracy
deviation = (results_df['accuracy'] + results_df['error_rate'] - 100).abs().max()
print(f"Max |accuracy + error_rate - 100|: {deviation:.2e}")

# 2. Cumulative explained variance from a full PCA fitted on training data only
full_pca = PCA(random_state=42).fit(X_train)
cumulative = np.cumsum(full_pca.explained_variance_ratio_) * 100
for n, reported in zip(results_df['n_components'], results_df['explained_variance']):
    print(f"  {n:>3} components: CSV {reported:7.3f}%  recomputed {cumulative[n - 1]:7.3f}%")


# 3. Re-run 30 components and the baseline, then compare the predictions
def predict(X_tr, X_te):
    knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, metric='euclidean')
    knn.fit(X_tr, y_train)
    return knn.predict(X_te)


pca = PCA(n_components=30, random_state=42).fit(X_train)
pred_30 = predict(pca.transform(X_train), pca.transform(X_test))
pred_784 = predict(X_train, X_test)

print(f"\nAccuracy (30 components): {np.mean(pred_30 == y_test) * 100:.2f}%")
print(f"Accuracy (784 baseline):  {np.mean(pred_784 == y_test) * 100:.2f}%")

# McNemar test on the discordant pairs (exact binomial form)
only_30_correct = int(np.sum((pred_30 == y_test) & (pred_784 != y_test)))
only_784_correct = int(np.sum((pred_30 != y_test) & (pred_784 == y_test)))
p_value = binomtest(min(only_30_correct, only_784_correct),
                    only_30_correct + only_784_correct).pvalue
print(f"Correct only with 30 PCs: {only_30_correct}")
print(f"Correct only with 784-D:  {only_784_correct}")
print(f"Exact McNemar p-value:    {p_value:.2e}")
