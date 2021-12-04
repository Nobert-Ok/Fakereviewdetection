import re
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model 
import streamlit as st
import pickle
import tensorflow as tf



model = load_model("model.pkl") 
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

label2int = {'0': 0, '1': 1}
int2label = {0: '0', 1: '1'}
SEQUENCE_LENGTH = 100 

def preprocess_text(sen):
    sentence = remove_tags(sen)
    sentence = re.sub('[^a-zA-Z]', ' ', str(sentence))
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', str(sentence))
    sentence = re.sub(r'\s+', ' ', str(sentence))

    return sentence

TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', str(text))

def get_predictions(text):
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    # pad the sequence
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    # get the prediction
    prediction = model.predict(sequence)[0]
    # one-hot encoded vector, revert using np.argmax
    return int2label[np.argmax(prediction)]


# def show_predict(email_text):
#     prediction = get_predictions(email_text)
#     email_type = ""
#     if(prediction == '1'):
#         email_type = "The prediction is "+str(prediction)+" It is a Spam"
#     else:
#         email_type = "The prediction is "+str(prediction)+" It is not a Spam"
    
#     return email_type

st.title("Fake/Real Review Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. predict
    result = get_predictions(input_sms)
    # 2. Display
    if result == 1:
        st.header("Fake Review")
    else:
        st.header("Genuine Review")

