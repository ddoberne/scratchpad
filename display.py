import streamlit as st
from transformers import pipeline

@st.cache
def load_generator():
  generator = pipeline('text-generation', model = 'gpt2')
  return generator

@st.cache
def load_classifier():
  classifier = pipeline('sentiment-analysis', model = 'distilbert-base-uncased-finetuned-sst-2-english')

classifier = load_classifier()
generator = load_generator()

st.title('How are you feeling today?')
s = st.text_input(label = 'I am feeling...')
response = None
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
  
if response == ':slightly_frowning_face:':
  if st.button(label = 'Turn that frown upside down!'):
    while output['label'] == 'NEGATIVE':
      benchmark = output['score']
      generates = generator(s, max_new_tokens = 15, num_return_sequences = 5, return_full_text = False)
      for generate in generates:
        generate = generate['generated_text']
        test_output = classifier(s + generate)[0]
        if (test_output['label'] == 'POSITIVE') or (test_output['score'] < output['score']):
          s = s + generate
          output = test_output
          break
      st.write(s)
      
