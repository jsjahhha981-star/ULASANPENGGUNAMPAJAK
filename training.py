# ==========================================
# IMPORT LIBRARY
# ==========================================
import pandas as pd
import re
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# LOAD DATA
# ==========================================
review = pd.read_csv("hasil_pre_process.csv")

# ==========================================
# CLEANING
# ==========================================
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

review['clean'] = review['Normalized'].apply(clean_text)

# ==========================================
# DATA
# ==========================================
X = review['clean']
Y = review['sentimen']

# ==========================================
# SPLIT DATA
# ==========================================
x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)
# ==========================================
# SAVE DATA TEST
# ==========================================
test_data = pd.DataFrame({
    "content": x_test,
    "sentimen": y_test
})

test_data.to_csv("test_data.csv", index=False)

print("DATA TEST BERHASIL DISIMPAN")
# ==========================================
# TF-IDF
# ==========================================
tfidf = TfidfVectorizer()

x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)

print("TF-IDF BERHASIL FIT")

# ==========================================
# SVM
# ==========================================
svm_model = SVC(kernel='rbf', probability=True)

svm_model.fit(x_train_tfidf, y_train)

print("SVM BERHASIL TRAINING")

# ==========================================
# PREDIKSI SVM
# ==========================================
y_pred = svm_model.predict(x_test_tfidf)

# ==========================================
# EVALUASI
# ==========================================
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(acc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# CONFUSION MATRIX
# ==========================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ==========================================
# K-MEANS
# ==========================================
kmeans_model = KMeans(
    n_clusters=19,
    random_state=42,
    n_init=10
)

kmeans_model.fit(x_train_tfidf)

print("K-MEANS BERHASIL TRAINING")

# ==========================================
# TEST SVM
# ==========================================
test = tfidf.transform(["aplikasi sangat bagus"])

hasil = svm_model.predict(test)

print("\nHASIL TEST:")
print(hasil)

# ==========================================
# SAVE MODEL
# ==========================================
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(kmeans_model, "kmeans_model.pkl")
joblib.dump(tfidf, "tfidf.pkl")

print("\nSEMUA MODEL BERHASIL DISIMPAN")