import streamlit as st
import numpy as np
import re
from tensorflow.keras.models import load_model

model = load_model('model')
def simple_preprocessing(text):
    emoji_pattern = re.compile("["
                               "\U0001F600-\U0001F64F"
                               "\U0001F300-\U0001F5FF"
                               "\U0001F680-\U0001F6FF"
                               "\U0001F1E0-\U0001F1FF"
                               "\U00002702-\U000027B0"
                               "\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'[^\w\s_]', '', text)
    text = " ".join(text.split())
    text = text.strip().lower()
    return text

st.title('Sentiment Analysis Project')
st.write('45 words max')
x = st.text_input('Input text: ')
x = simple_preprocessing(x)
x = np.array([x])
y = model.predict(x)
y = y[0][0]
if y <= 0.4:
  sentiment = 'Tiêu cực'
if y >= 0.6:
  sentiment = 'Tích cực'
else:
  sentiment = 'Trung lập'

if st.button('Xử lý'):
  st.write('Hàm ý:', sentiment)
