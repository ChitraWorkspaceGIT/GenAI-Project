import streamlit as st
import joblib 

vectorizer = joblib.load('vectorizer.jb')
model = joblib.load('lr_model.jb')

st.title("Fake news detection with LLM Classification")
st.write("Enter the news to check if it is real or no")

news_input = st.text_area("News Article:","")
if st.button("Predict"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)
        if prediction[0] == 1:
            st.success("This news is real")
        else:
            st.error("This news is fake")
    else:
        st.warning("Please enter a news article to check.")