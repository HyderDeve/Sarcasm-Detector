import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


model = load_model('Urdu_Sarcasm_Detector.keras')

def text_preprocessing(sentence):
    
    stopwords = []

    with open('stopwords.txt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            stopwords.append(line.strip())

    urdu_punctuation = "؛۔،؟!‘’“”()-"
    table = str.maketrans('','',urdu_punctuation)

    # Process sentences to remove stopwords and punctuation
    processed_sentences = []
    sentence = sentence.lower()
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    soup = BeautifulSoup(sentence, features="lxml")
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table) #removal of punctutaion in the sentence
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " " #removal of stopwords
    processed_sentences.append(filtered_sentence)

    sentences = pd.Series(processed_sentences)
    return sentences

def tokenization_and_padding(sentences):
    
    vocab_size = 10000
    tokenizer = Tokenizer(num_words=vocab_size, oov_token = '<OOV>')

    tokenizer.fit_on_texts(sentences)
    word_idx = tokenizer.word_index

    st.text(f'The Word Index List For This Data Is: \n {word_idx}')

    sequences = tokenizer.texts_to_sequences(sentences)

    max_length = 15

    padded_sequence = pad_sequences(maxlen = max_length, sequences = sequences, padding = 'post', truncating = 'post')

    return padded_sequence


st.title('Urdu Sentences Sarcasm Detector')


sentences = st.text_input(label= 'Enter Urdu Sentence', placeholder = 'اس طرح سے۔', help = 'Enter Urdu Sentences Here')

if st.button('Check Sentiment'):
    if sentences:
        sentence = text_preprocessing(sentences)

        padded_sequence = tokenization_and_padding(sentence)

        padded_sequence = np.array(padded_sequence)

        predictions = model.predict(padded_sequence)


        for inference in predictions:
            if inference > 0.5:
                st.text('Negative Sentiment', width='stretch')
            else:
                st.text('Positive Sentiment', width='stretch')