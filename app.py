import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, jsonify
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)

vectorizer = joblib.load('vectorizer_opti.pkl')
model = tf.keras.models.load_model('my_model_opti.h5')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        
        # Transform the input text using the vectorizer
        text_vectorized = vectorizer.transform([text])
        
        # Convert the sparse matrix to a dense matrix if needed
        text_vectorized = text_vectorized.todense()

        # Perform prediction
        prediction = model.predict(text_vectorized)
        # Get the probability of the positive class
        positive_prob = prediction[0][1]
        sentiment = 'Positive' if positive_prob > 0.5 else 'Negative'
        confidence = positive_prob if sentiment == 'Positive' else 1 - positive_prob

        # Convert confidence to percentage
        confidence_percentage = round(confidence * 100, 2)
        
        return render_template('index.html', prediction=sentiment, text=text, confidence_percentage=confidence_percentage)
    
    return render_template('index.html', prediction=None, text=None, confidence_percentage=None)

