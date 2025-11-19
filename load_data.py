# -------------------------
# retrain_essay_model.py
# -------------------------

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv('ann_sample.csv')  # Replace with your dataset path
essays = df['essay_text'].astype(str)
scores = df['score'].values

# -------------------------
# Feature Extraction
# -------------------------
# 1️⃣ TF-IDF features
vectorizer = TfidfVectorizer(max_features=500)
tfidf_features = vectorizer.fit_transform(essays).toarray()

# 2️⃣ Numeric features
word_counts = np.array([len(e.split()) for e in essays]).reshape(-1, 1)
avg_word_lengths = np.array([np.mean([len(w) for w in e.split()]) if len(e.split())>0 else 0 for e in essays]).reshape(-1,1)
sentence_counts = np.array([e.count('.') + e.count('!') + e.count('?') for e in essays]).reshape(-1,1)
punctuation_counts = np.array([sum(1 for c in e if c in '.,;!?') for e in essays]).reshape(-1,1)

numeric_features = np.hstack([word_counts, avg_word_lengths, sentence_counts, punctuation_counts])

# Combine all features
X = np.hstack([tfidf_features, numeric_features])
y = scores

# -------------------------
# Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# Build ANN Model
# -------------------------
model = Sequential([
    Dense(128, input_dim=X.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# -------------------------
# Train Model
# -------------------------
print("Training model...")
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# -------------------------
# Save Model & Vectorizer
# -------------------------
model.save('essay_model_updated.h5')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully!")
