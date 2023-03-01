import streamlit as st
import tensorflow as tf
from keras_models.Preprocess import Preprocessing

# @st.cache_resource
def load_model_local(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def preload_model(attributes=['cAGR', 'cCON', 'cEXT', 'cOPN', 'cNEU'], model_no=1,dir_path='./checkpoint'):
    preloaded = []
    for attrbute in attributes:
        model_path = f'{dir_path}/{attrbute}_model_{model_no}.tf'
        preloaded.append(load_model_local(model_path))
    return preloaded

def preprocess(text):
    prep = Preprocessing()
    prep.load_text(text)
    return prep
def predict(model):
    return model.predict(textObj.X, verbose=2)[0]

A, C, E, O, N = preload_model(model_no=1,dir_path='./checkpoint')
# model = load_model_local('./checkpoint/cAGR_model_1.tf')
textObj = None

st.title("Text-based Five-Factor Model Personality Prediction with BiLSTM")
st.write("Created by Nicholas Yan for the 2023 Swire Hotel Hackathon")
text = st.text_area(' ', placeholder='Enter text here to analyze ...')
if st.button('Predict') or text:
    if not textObj:
        st.write(text)
        textObj = preprocess(text)
    st.write('Predicted personality traits:')
    st.write('A',predict(A))
    st.write('C',predict(C))
    st.write('E',predict(E))
    st.write('O',predict(O))
    st.write('N',predict(N))




