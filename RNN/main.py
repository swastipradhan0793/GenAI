### Load all libraries 
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# mapping of words index back to word
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

## load the pre-trained model with ReLu activation
model = load_model('simple_rnn_imdb.h5')



# Helper function
# function to decode review

def decoded_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

# function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


## Prediction function

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment, prediction[0][0]


### Steamlit app
import streamlit as st

st.title('IMDM Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify as positive or negative')

#User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    # preprocess_input = preprocess_text(user_input)

    # prediction
    # prediction = model.predict(preprocess_input)
    sentiment, score = predict_sentiment(user_input)

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {score}')
else:
    st.write(f'Please enter a movie review')