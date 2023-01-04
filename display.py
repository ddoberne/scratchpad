import streamlit as st
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

st.title('How are you feeling today?')
s = st.text_input(label = 'I am feeling...')
if s:
  output = classifier(s)[0]
  if output['score'] > 0.75:
    if output['label'] == 'POSITIVE':
      response = ':slightly_smiling_face:'
    else:
      response = ':slightly_frowning_face:'
  else:
    response = ':neutral_face:'
  st.title(response)
