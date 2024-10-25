import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = tf.keras.models.load_model('fake_news_model.h5')

# Load the tokenizer used during training
# You should save the tokenizer after training, here we are assuming it's saved as 'tokenizer.pickle'.
import pickle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Streamlit app
st.title("Fake News Detection")

# User input
user_input = st.text_input("Enter news headline:")

if st.button("Predict"):
    # Preprocess the user input
    input_sequence = tokenizer.texts_to_sequences([user_input])
    input_padded = pad_sequences(input_sequence, padding='post', maxlen=100)  # Same maxlen as used during training

    # Make prediction
    prediction = model.predict(input_padded)
    label = 'FAKE' if prediction[0][0] < 0.5 else 'REAL'
    
    # Display the result
    st.write(f"The news headline is: **{label}**")


