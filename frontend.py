import streamlit as st

st.title("Sentiment Analysis")
text = st.text_area("Enter your thoughts...")

fetch_results = st.button("Check Sentiment")

if fetch_results:
    try:
        if text == "":
            raise ValueError("Empty text")
        sentiment = infer_sentiment(text)
        st.write("Predicted Sentiment: {}".format(sentiment))
        
    except ValueError:
        st.write("Can't make predictions on empty text, can we? 🤔")
    except:
        st.write("Unable to reach the API at the moment...😞")