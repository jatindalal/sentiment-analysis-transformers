from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from preprocessor import Preprocessor
import torch
import streamlit as st

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_name = "hazardous/bert_base_uncased_finetuned_imdb"
preprocessor = Preprocessor()

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return model.to(device), tokenizer

model, tokenizer = load_model()

@st.cache_data
def infer_sentiment(sentence):
    preprocessed_sentence = preprocessor.preprocess([sentence])[0]
    
    inputs = tokenizer(preprocessed_sentence, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs.to(device)).logits
    predicted_class_id = logits.argmax().item()

    return preprocessed_sentence, model.config.id2label[predicted_class_id]

st.title("Sentiment Analysis")
text = st.text_area("Enter your thoughts...")

fetch_results = st.button("Check Sentiment")

if fetch_results:
    try:
        if text == "":
            raise ValueError("Empty text")
        preprocessed_sentence, sentiment = infer_sentiment(text)
        sentiment = 'Negative' if sentiment == 'neg' else 'Positive'
        st.write("Preprocessed Text: {}".format(preprocessed_sentence))
        st.write("Predicted Sentiment: {}".format(sentiment))
        
    except ValueError:
        st.write("Can't make predictions on empty text, can we? 🤔")
    except:
        st.write("Something went wrong...😞")
