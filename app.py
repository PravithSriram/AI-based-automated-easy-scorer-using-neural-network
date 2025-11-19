from flask import Flask, render_template, request
from textblob import TextBlob
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# -------------------------
# Load Model and Vectorizer
# -------------------------
# Load ANN model without compiling to avoid 'mse' error
model = tf.keras.models.load_model('essay_model_updated.h5', compile=False)
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Set the MAX_SCORE based on training dataset maximum score
MAX_MODEL_SCORE = 150  # <-- adjust based on your dataset

# -------------------------
# Grammar, Spelling, Sentence Structure, and Punctuation
# -------------------------
def grammar_score(text):
    blob = TextBlob(text)
    total_sentences = len(blob.sentences)
    if total_sentences == 0:
        return 10  # avoid divide by zero

    spelling_errors = 0
    sentence_errors = 0
    punctuation_errors = 0

    for sentence in blob.sentences:
        corrected = sentence.correct()
        if sentence != corrected:
            spelling_errors += 1

        # Check for sentence structure: very short or very long sentences
        if len(sentence.split()) < 3 or len(sentence.split()) > 30:
            sentence_errors += 1

        # Check missing punctuation at the end
        if sentence.raw.strip()[-1] not in ".!?":
            punctuation_errors += 1

    # Weighted scoring
    spelling_score = 1 - (spelling_errors / total_sentences)
    structure_score = 1 - (sentence_errors / total_sentences)
    punctuation_score = 1 - (punctuation_errors / total_sentences)

    # Combine all grammar-related scores (0-10 scale)
    grammar_total = (spelling_score + structure_score + punctuation_score) / 3
    return round(grammar_total * 10, 2)

# -------------------------
# Predict Essay Score
# -------------------------
def predict_essay_score(essay):
    # 1️⃣ Extract numeric features
    word_count = len(essay.split())
    avg_word_length = np.mean([len(w) for w in essay.split()]) if word_count > 0 else 0
    sentence_count = essay.count('.') + essay.count('!') + essay.count('?')
    punctuation_count = sum(1 for c in essay if c in '.,;!?')

    numeric_features = np.array([[word_count, avg_word_length, sentence_count, punctuation_count]])

    # 2️⃣ TF-IDF features
    tfidf_features = vectorizer.transform([essay]).toarray()

    # 3️⃣ Combine features
    features = np.hstack([tfidf_features, numeric_features])

    # 4️⃣ Predict using ANN model
    try:
        raw_model_score = float(model.predict(features)[0][0])
    except:
        raw_model_score = 0

    # Normalize model score to 0-10 scale
    model_score = (raw_model_score / MAX_MODEL_SCORE) * 10
    model_score = max(0, min(10, model_score))

    # 5️⃣ Calculate grammar
    grammar = grammar_score(essay)

    # 6️⃣ Weighted final score
    final_score = 0.7 * model_score + 0.3 * grammar
    final_score = max(0, min(10, round(final_score, 2)))  # clamp 0-10

    return final_score, grammar, round(model_score, 2)

# -------------------------
# Flask Route
# -------------------------
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        essay = request.form['essay']
        final_score, grammar, model_score = predict_essay_score(essay)
        return render_template('index.html',
                               essay=essay,
                               score=final_score,
                               grammar=grammar,
                               model=model_score)
    return render_template('index.html')

# -------------------------
# Run App
# -------------------------
if __name__ == '__main__':
    app.run(debug=True)
